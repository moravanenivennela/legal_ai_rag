import re

with open("rag_engine.py", "r", encoding="utf-8") as f:
    content = f.read()

old_method = '''    def get_topic_image(self, predicted_domain: str) -> str:
        """Maps a detected legal topic to a representative image from the custom-trained image classifier's categories."""
        import json
        topic_map = {
            "constitution": "government_building",
            "consumer_protection": "shopping_consumer",
        }
        image_category = topic_map.get(predicted_domain)
        if not image_category:
            return None
        try:
            with open("class_sample_images.json", "r") as f:
                samples = json.load(f)
            return samples.get(image_category)
        except Exception:
            return None'''

new_method = '''    def get_topic_image(self, predicted_domain: str, shown_images: set) -> str:
        """Returns a random, not-yet-shown image for the detected topic. Resets the pool if exhausted."""
        import json
        import random
        topic_map = {
            "constitution": "government_building",
            "consumer_protection": "shopping_consumer",
        }
        image_category = topic_map.get(predicted_domain)
        if not image_category:
            return None
        try:
            with open("class_all_images.json", "r") as f:
                manifest = json.load(f)
            all_images = manifest.get(image_category, [])
            if not all_images:
                return None
            unseen = [img for img in all_images if img not in shown_images]
            if not unseen:
                unseen = all_images  # pool exhausted, reset
            return random.choice(unseen)
        except Exception:
            return None'''

if old_method in content:
    content = content.replace(old_method, new_method)
    print("Replaced get_topic_image successfully.")
else:
    print("WARNING: old method text not found, no changes made. Manual edit needed.")

with open("rag_engine.py", "w", encoding="utf-8") as f:
    f.write(content)
