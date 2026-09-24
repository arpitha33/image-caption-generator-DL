from gtts import gTTS


def text_to_speech(text, output_file="caption.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(output_file)

    print(f"Audio saved to: {output_file}")


if __name__ == "__main__":

    caption = input("Enter caption: ").strip()

    if not caption:
        print("Please enter a caption.")
        exit()

    text_to_speech(caption)