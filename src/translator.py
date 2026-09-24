from deep_translator import MyMemoryTranslator


def translate_text(text, target_language="english"):
    try:
        translated = MyMemoryTranslator(
            source="english",
            target=target_language
        ).translate(text)

        return translated

    except Exception as e:
        print("Translation failed:", e)
        return None


if __name__ == "__main__":

    caption = input("Enter caption: ").strip()

    if not caption:
        print("Please enter a caption.")
        exit()

    print("\nAvailable languages:")
    print("1. English")
    print("2. Hindi")
    print("3. Kannada")
    print("4. Tamil")
    print("5. Telugu")
    print("6. Malayalam")
    print("7. Spanish")
    print("8. French")
    print("9. German")

    choice = input(
        "\nChoose translation language (default: English): "
    ).strip().lower()

    languages = {
        "1": "english",
        "2": "hindi",
        "3": "kannada",
        "4": "tamil",
        "5": "telugu",
        "6": "malayalam",
        "7": "spanish",
        "8": "french",
        "9": "german",

        "english": "english",
        "hindi": "hindi",
        "kannada": "kannada",
        "tamil": "tamil",
        "telugu": "telugu",
        "malayalam": "malayalam",
        "spanish": "spanish",
        "french": "french",
        "german": "german"
    }

    if choice == "":
        target_language = "english"
    elif choice in languages:
        target_language = languages[choice]
    else:
        print("Invalid choice. Using English.")
        target_language = "english"

    if target_language == "english":
        translated = caption
    else:
        translated = translate_text(caption, target_language)

    if translated:
        print("\nOriginal caption:", caption)
        print("Translated caption:", translated)