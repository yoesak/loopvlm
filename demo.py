import itertools
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import transformers
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
import os
import string
import functools
import re
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import cv2
from model import Transformer
from quantize import WeightOnlyInt8QuantHandler
from pathlib import Path
from typing import Optional, Tuple
import time
import contextlib



model_id = "google/paligemma-3b-mix-224"
processor_id = "google/paligemma-3b-mix-224"
checkpoint_path = Path("checkpoints/google/paligemma-3b-mix-224/model_int8.pth")



COLORS = ['#03256C', '#2541B2', '#1768AC', '#06BEE1', '#FFFFFF']
device = "cuda:0"
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        checkpoint_path.parent,
        torch_dtype=torch.bfloat16,
        device_map=device,
        revision="bfloat16",
    ).eval()

processor = PaliGemmaProcessor.from_pretrained(processor_id)
use_tp = False
precision = torch.bfloat16

vision_model = base_model.vision_tower
projector = base_model.multi_modal_projector


max_new_tokens = 10
speculate_k = 5
temperature = 0.0
top_k = 200

###### Load Quantized Model, only support int8 quantized model, please use scrips/prepare.sh to quantize a model
def _load_model():
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    ### nt8 weight-only quantization
    simple_quantizer = WeightOnlyInt8QuantHandler(model)
    model = simple_quantizer.convert_for_runtime()
   
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=torch.bfloat16)

    return model.eval()

model = _load_model()

torch._dynamo.config.capture_scalar_outputs = True

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    #logits[0, -1, 1] = -10

    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    idx_next = torch.tensor([torch.argmax(logits[0, -1])]).to('cuda:0')

    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, embeds: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, embeds=embeds)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs

def model_forward(model, x, input_pos):
    return model(x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    embeds: torch.Tensor,
    max_new_tokens: int,
    
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    print("prefill")
    next_token = prefill(model, prompt.view(1, -1), embeds, input_pos, **sampling_kwargs)
    if is_speculative:
        prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[T + 1:] = torch.cat(generated_tokens)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device='cuda'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def parse_location(input_string):
    pattern = r'((<(?:loc\d+>))+) ([^<]+)'
    matches = re.findall(pattern, input_string)
    return matches
    

def convert_bbox(bbox, original_size=(1024, 1024), target_size=(480, 854)):
    """
    Convert bounding box coordinates from the original resolution to the target resolution.

    Parameters:
    bbox (tuple): A tuple (x1, y1, x2, y2) representing the bounding box coordinates in the original resolution.
    original_size (tuple): A tuple (width, height) representing the original resolution.
    target_size (tuple): A tuple (width, height) representing the target resolution.

    Returns:
    tuple: A tuple (x1, y1, x2, y2) representing the bounding box coordinates in the target resolution.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size

    x1, y1, x2, y2 = bbox

    x1 = int(x1 * target_width / original_width)
    y1 = int(y1 * target_height / original_height)
    x2 = int(x2 * target_width / original_width)
    y2 = int(y2 * target_height / original_height)

    return (x1, y1, x2, y2)



###### Transformers Inference
def infer(
    image: Image.Image,
    text: str,
    max_new_tokens: int
) -> str:
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
      generated_ids = model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=False
      )
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return result[0][len(text):].lstrip("\n")


##### Parse segmentation output tokens into masks
##### Also returns bounding boxes with their labels
def parse_segmentation(input_image: Image.Image, input_text: str):
  
  out = infer(input_image, input_text, max_new_tokens=max_new_tokens)
  print(out)

  objs = extract_objs(out.lstrip("\n"), input_image.size[0], input_image.size[1], unique_labels=True)
  
  labels = set(obj.get('name') for obj in objs if obj.get('name'))
  color_map = {l: COLORS[i % len(COLORS)] for i, l in enumerate(labels)}
  
  highlighted_text = [(obj['content'], obj.get('name')) for obj in objs]
  
  annotated_img = (
    input_image,
    [
        (
            obj['mask'] if obj.get('mask') is not None else obj['xyxy'],
            obj['name'] or '',
        )
        for obj in objs
        if 'mask' in obj or 'xyxy' in obj
    ],
  )
  has_annotations = bool(annotated_img[1])
  return (labels, annotated_img)

def vid_parse_segmentation(vid_path: str, vid_start: int, vid_end: int, prompt: str):
  cap = cv2.VideoCapture(vid_path)
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  frames = []
  
  last_frame = None

  for i in range(int(fps * vid_end)):
    ret, frame = cap.read()

    if i > int(fps * vid_start):
        if not ret:
            break
        last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(last_frame)
        frames.append(pil_frame)

  cap.release()

  # Write output_video first
  out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (last_frame.shape[1], last_frame.shape[0]))

  model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])

  global decode_one_token, prefill
  decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

  # Uncomment to squeeze more perf out of prefill
  prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

  aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
  }
  
  start = -1
  embed = model.get_tok_embeddings()

  bounding_boxes = []
  model_fps = 16

  for i, frame in enumerate(frames):
    if i % 2== 0:
        model_inputs = processor(text=prompt, images=frame, return_tensors="pt").to('cuda:0')
        encoded = model_inputs['input_ids'][0]
        prompt_length = encoded.size(0)

        embedding_values = embed(encoded)

        img_embed = projector(vision_model(model_inputs.pixel_values.to(dtype=torch.bfloat16)).last_hidden_state)

        img_embed = img_embed / (2048 ** 0.5)
        embedding_values[:256, :] = img_embed[0]
        embedding_values = embedding_values.unsqueeze(0)

        device_sync(device=device) # MKG
 
        callback = lambda x : x
        t0 = time.perf_counter()
        prof = contextlib.nullcontext()
    
        with prof:
            y, metrics = generate(
                model,
                encoded,
                embedding_values,
                max_new_tokens,
                draft_model=None,
                speculate_k=speculate_k,
                interactive=False,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue

        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        print(processor.decode(y, skip_special_tokens=True))

        decoded_output = processor.decode(y, skip_special_tokens=True)
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)

        new_model_fps = int(1 / t)
        if new_model_fps != model_fps:
            model_fps=new_model_fps
        print(f"Model fps {new_model_fps}")
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        print(processor.decode(y, skip_special_tokens=True))

        bounding_note = ""
        if ('loc' in decoded_output):
            matches = parse_location(decoded_output)
            for i, (coordinates, last_coordinate, description) in enumerate(matches, 1):
                bounding_boxes = []
                locations = []
                locations = [int(re.sub(r'[A-Za-z\n<>]', '', loc)) for loc in coordinates.split("><") if 'loc' in loc]
                bounding_boxes.append(locations[0])
                bounding_boxes.append(locations[1])
                bounding_boxes.append(locations[2])
                bounding_boxes.append(locations[3])
                
                bounding_boxes = convert_bbox(bounding_boxes)
                bounding_boxes = [bounding_boxes[1], bounding_boxes[0], bounding_boxes[3], bounding_boxes[2]]

                draw = ImageDraw.Draw(frame)

                draw.rectangle(bounding_boxes, outline="lime", width=3)
                text_position = (bounding_boxes[2] - 5, bounding_boxes[3] - 5)
                draw.text(text_position, description, fill="lime", font_size=30)           

    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    out.write(frame)

  return "output_video.mp4"



######## Demo

INTRO_TEXT = """## Boring Task AI Vision demo\n\n
| DEMO 2 - Video Vision
\n\n
**This is an experimental research model.** Make sure to add appropriate guardrails when using the model for applications.
"""

custom_css = """
#custom-blocks {
    max-width: 640px;
    margin: 0 auto; /* Center the block */
}
"""

with gr.Blocks(css=custom_css, theme="soft") as demo:
  with gr.Column(elem_id="custom-blocks"):
    gr.Markdown(INTRO_TEXT)
    with gr.Tab("Detect"):
      vid_path = gr.Video(label="Input")

      with gr.Row():
          vid_start = gr.Text(label="Start video")
          vid_end = gr.Text(label="End video")
        
      prompt = gr.Text(label="Prompt")
      
      seg_btn = gr.Button("Submit")

      annotated_video = gr.Video(label="Output")

    seg_inputs = [
        vid_path,
        vid_start,
        vid_end,
        prompt
    ]
    
    seg_outputs = [
        annotated_video
    ]
    
    seg_btn.click(
        fn=vid_parse_segmentation,
        inputs=seg_inputs,
        outputs=seg_outputs,
    )


### Postprocessing Utils for Segmentation Tokens
### Segmentation tokens are passed to another VAE which decodes them to a mask

_MODEL_PATH = 'vae-oid.npz'

_SEGMENT_DETECT_RE = re.compile(
    r'(.*?)' +
    r'<loc(\d{4})>' * 4 + r'\s*' +
    '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
    r'\s*([^;<>]+)? ?(?:; )?',
)


def _get_params(checkpoint):
  """Converts PyTorch checkpoint to Flax params."""

  def transp(kernel):
    return np.transpose(kernel, (2, 3, 1, 0))

  def conv(name):
    return {
        'bias': checkpoint[name + '.bias'],
        'kernel': transp(checkpoint[name + '.weight']),
    }

  def resblock(name):
    return {
        'Conv_0': conv(name + '.0'),
        'Conv_1': conv(name + '.2'),
        'Conv_2': conv(name + '.4'),
    }

  return {
      '_embeddings': checkpoint['_vq_vae._embedding'],
      'Conv_0': conv('decoder.0'),
      'ResBlock_0': resblock('decoder.2.net'),
      'ResBlock_1': resblock('decoder.3.net'),
      'ConvTranspose_0': conv('decoder.4'),
      'ConvTranspose_1': conv('decoder.6'),
      'ConvTranspose_2': conv('decoder.8'),
      'ConvTranspose_3': conv('decoder.10'),
      'Conv_1': conv('decoder.12'),
  }


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
  batch_size, num_tokens = codebook_indices.shape
  assert num_tokens == 16, codebook_indices.shape
  unused_num_embeddings, embedding_dim = embeddings.shape

  encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
  encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
  return encodings


@functools.cache
def _get_reconstruct_masks():
  """Reconstructs masks from codebook indices.
  Returns:
    A function that expects indices shaped `[B, 16]` of dtype int32, each
    ranging from 0 to 127 (inclusive), and that returns a decoded masks sized
    `[B, 64, 64, 1]`, of dtype float32, in range [-1, 1].
  """

  class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
      original_x = x
      x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
      x = nn.relu(x)
      x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
      x = nn.relu(x)
      x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
      return x + original_x

  class Decoder(nn.Module):
    """Upscales quantized vectors to mask."""

    @nn.compact
    def __call__(self, x):
      num_res_blocks = 2
      dim = 128
      num_upsample_layers = 4

      x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
      x = nn.relu(x)

      for _ in range(num_res_blocks):
        x = ResBlock(features=dim)(x)

      for _ in range(num_upsample_layers):
        x = nn.ConvTranspose(
            features=dim,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=2,
            transpose_kernel=True,
        )(x)
        x = nn.relu(x)
        dim //= 2

      x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)

      return x

  def reconstruct_masks(codebook_indices):
    quantized = _quantized_values_from_codebook_indices(
        codebook_indices, params['_embeddings']
    )
    return Decoder().apply({'params': params}, quantized)

  with open(_MODEL_PATH, 'rb') as f:
    params = _get_params(dict(np.load(f)))

  return jax.jit(reconstruct_masks, backend='cpu')

def extract_objs(text, width, height, unique_labels=False):
  """Returns objs for a string with "<loc>" and "<seg>" tokens."""
  objs = []
  seen = set()
  while text:
    m = _SEGMENT_DETECT_RE.match(text)
    if not m:
      break
    print("m", m)
    gs = list(m.groups())
    before = gs.pop(0)
    name = gs.pop()
    y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
    
    y1, x1, y2, x2 = map(round, (y1*height, x1*width, y2*height, x2*width))
    seg_indices = gs[4:20]
    if seg_indices[0] is None:
      mask = None
    else:
      seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)
      m64, = _get_reconstruct_masks()(seg_indices[None])[..., 0]
      m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
      m64 = Image.fromarray((m64 * 255).astype('uint8'))
      mask = np.zeros([height, width])
      if y2 > y1 and x2 > x1:
        mask[y1:y2, x1:x2] = np.array(m64.resize([x2 - x1, y2 - y1])) / 255.0

    content = m.group()
    if before:
      objs.append(dict(content=before))
      content = content[len(before):]
    while unique_labels and name in seen:
      name = (name or '') + "'"
    seen.add(name)
    objs.append(dict(
        content=content, xyxy=(x1, y1, x2, y2), mask=mask, name=name))
    text = text[len(before) + len(content):]

  if text:
    objs.append(dict(content=text))

  return objs

#########

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")