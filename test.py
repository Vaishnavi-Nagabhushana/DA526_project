import argparse
import os
import torch
from pathlib import Path
from PIL import Image  
from torchvision import transforms
from torchvision.utils import save_image
from models.StyTR import StyTrans, decoder # corrected import clearly explicitly

# Test-time image transformation explicitly
def test_transform(size=512):
    transform_list = [
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def main(args):
    device = torch.device(args.device)

    # Define output explicitly clearly
    os.makedirs(args.output, exist_ok=True)

    # explicitly corrected initialization (aligned with training clearly)
    with torch.no_grad():
        network = StyTrans(decoder=decoder, args=args)
    network.eval().to(device)

    # Loading your trained checkpoint explicitly and clearly explicitly
    state_dict = torch.load(args.decoder_path, map_location=device)
    network.load_state_dict(state_dict, strict=False)

    # Preprocessing clearly
    content_tf = test_transform(size=args.image_size)
    style_tf = test_transform(size=args.image_size)

    # Listing paired images explicitly clearly
    content_paths = sorted(list(Path(args.content_dir).glob('*')))
    style_paths = sorted(list(Path(args.style_dir).glob('*')))

    # Check clearly if the number of images match explicitly
    assert len(content_paths) == len(style_paths), \
           f"Number of content images ({len(content_paths)}) and style images ({len(style_paths)}) differ!"

    for content_path, style_path in zip(content_paths, style_paths):
        print(f"\n💡 Processing pair:\n Content: {content_path.name}\n Style: {style_path.name}")

        content_image = content_tf(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        style_image = style_tf(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

        # Generate stylized image clearly explicitly
        with torch.no_grad():
            output, _, _, _, _ = network(content_image, style_image)

        # Save explicitly stylized result clearly explicitly
        output_filename = os.path.join(args.output, f"{content_path.stem}_stylized_{style_path.stem}.jpg")
        save_image(output.cpu(), output_filename, normalize=True)
        
        print(f"✅ Stylized image saved clearly at: {output_filename}")

    print("\n✅🎉 All testing completed successfully clearly!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the StyTr² network clearly & explicitly.")
    
    # Important clearly correct arguments explicitly clearly
    parser.add_argument('--content_dir', type=str, required=True, help='Directory path containing content images.')
    parser.add_argument('--style_dir', type=str, required=True, help='Directory path containing style images (ground-truth cartoons).')
    parser.add_argument('--output', type=str, default='./stylized_results', help='Output directory for stylized images.')
    parser.add_argument('--decoder_path', type=str, required=True, help='Path explicitly to your trained checkpoint (.pth).')
    parser.add_argument('--image_size', type=int, default=512, help='Image size explicitly clearly for testing clearly.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension explicitly used for transformer explicitly.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device explicitly explicitly for inference explicitly.')

    args = parser.parse_args()
    main(args)


# python test.py --content_dir "D:\ghibli_test\real" --style_dir "D:\ghibli_test\ghibli" --decoder_path "D:\DA526_Project\experiments\iter_1000.pth"