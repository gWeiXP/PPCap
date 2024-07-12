from clipscore.clipscore import *

def computer_clipscore_and_other(image_paths, model, device, candidates, references):
    # image_paths, str list, example:'example/images/image1.jpg'
    # candidates, generate caption, str list, example:'a cute cat'
    # references, refer caption, str list, example:'a cute cat'



    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)


    # get text-text clipscore
    _, per_instance_text_text = get_refonlyclipscore(
        model, references, candidate_feats, device)
    # F-score
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
    scores = {"clipscore":np.mean(per_instance_image_text), "refclipscores":np.mean(refclipscores)}

    other_metrics = generation_eval_utils.get_all_metrics(references, candidates)

    return scores, other_metrics


if __name__ == "__main__":
    image_dir = "example/images/"
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    candidates = ["an orange cat and a grey cat are lying together.", 
                  "a black dog wearing headphones looks at the camera as an orange cat walks in the background."]
    
    references = [
         ["two cats are sleeping next to each other.", "a grey cat is cuddling with an orange cat on a blanket.", "the orange cat is happy that the black cat is close to it."],
         ["a dog is wearing ear muffs as it lies on a carpet.", "a black dog and an orange cat are looking at the photographer.", "headphones are placed on a dogs ears."]
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-L/14", device=device, jit=False)
    model.eval()


    scores, other_metrics = computer_clipscore_and_other(image_paths, model, device, candidates, references)
    print(scores)
    print(other_metrics)