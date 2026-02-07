import torch
import gc
import argparse
import os
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

def main():
    # Argomenti da command line
    parser = argparse.ArgumentParser(description="Genera video con CogVideoX")
    parser.add_argument("prompt_file", help="Nome del file prompt (senza .txt) nella cartella prompts/")
    parser.add_argument("--model", type=str, default="2b", choices=["2b", "5b"], help="Modello da usare (default: 2b)")
    parser.add_argument("--frames", type=int, default=49, help="Numero di frame (default: 49)")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (default: 50)")
    parser.add_argument("--fps", type=int, default=8, help="FPS del video (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per riproducibilità (default: 42)")
    parser.add_argument("--guidance", type=float, default=6.0, help="Guidance scale per classifier-free guidance (default: 6.0)")
    parser.add_argument("--output", type=str, default=None, help="Percorso output personalizzato (default: outputs/<model>/<prompt>.mp4)")
    args = parser.parse_args()

    # Percorsi
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", f"{args.prompt_file}.txt")

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(script_dir, "outputs", args.model, f"{args.prompt_file}.mp4")

    # Crea cartella output se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Leggi il prompt dal file
    if not os.path.exists(prompt_path):
        print(f"Errore: File '{prompt_path}' non trovato!")
        print(f"\nPrompt disponibili:")
        prompts_dir = os.path.join(script_dir, "prompts")
        for f in os.listdir(prompts_dir):
            if f.endswith(".txt"):
                print(f"  - {f[:-4]}")
        return

    with open(prompt_path, "r") as f:
        prompt = f.read().strip()

    print(f"Prompt: {args.prompt_file}")
    print(f"Modello: CogVideoX-{args.model}")
    print(f"Frames: {args.frames} | Steps: {args.steps} | FPS: {args.fps} | Guidance: {args.guidance}")
    print("-" * 50)

    # Libera memoria prima di iniziare
    gc.collect()
    torch.cuda.empty_cache()

    model_id = f"THUDM/CogVideoX-{args.model}"
    print(f"Caricamento modello {model_id}...")

    # Il 5b richiede bfloat16, il 2b può usare float16
    dtype = torch.bfloat16 if args.model == "5b" else torch.float16

    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )

    # Ottimizzazioni per memoria limitata
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    print("Generazione video in corso...")
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=args.steps,
        num_frames=args.frames,
        guidance_scale=args.guidance,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]

    export_to_video(video, output_path, fps=args.fps)
    print(f"\nVideo salvato in: {output_path}")

if __name__ == "__main__":
    main()
