import { useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  Image as ImageIcon,
  Languages,
  RotateCcw,
  Sparkles,
  Upload,
  Volume2,
  Cpu,
} from "lucide-react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");

  const [model, setModel] = useState("attention");
  const [language, setLanguage] = useState("English");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [activeStep, setActiveStep] = useState(0);

  // Upload-area drag state
  const [uploadDragging, setUploadDragging] = useState(false);

  // Carousel drag state
  const [carouselDragging, setCarouselDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState(0);

  const dragStartX = useRef(0);
  const dragCurrentX = useRef(0);

  /* =========================================================
     IMAGE
  ========================================================= */

  const handleImage = (file) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
    setActiveStep(0);
    setDragOffset(0);
  };

  /* =========================================================
     GENERATE
  ========================================================= */

  const generateCaption = async () => {
    if (!image) {
      setError("Please upload an image first.");
      setActiveStep(0);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setActiveStep(2);

    try {
      const formData = new FormData();

      formData.append("image", image);
      formData.append("model", model);
      formData.append(
        "language",
        language.trim() || "English"
      );

      const response = await fetch(
        `${API_BASE}/generate`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Generation failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the backend."
      );

      setActiveStep(1);
    } finally {
      setLoading(false);
    }
  };

  /* =========================================================
     RESET
  ========================================================= */

  const resetAll = () => {
    setImage(null);
    setPreview("");
    setResult(null);
    setError("");
    setLanguage("English");
    setModel("attention");
    setActiveStep(0);
    setDragOffset(0);
    setCarouselDragging(false);
  };

  /* =========================================================
     CAROUSEL NAVIGATION
  ========================================================= */

  const goNext = () => {
    if (activeStep === 0 && !image) {
      setError("Please upload an image first.");
      return;
    }

    if (activeStep < 2) {
      setError("");
      setActiveStep((current) => current + 1);
    }
  };

  const goPrevious = () => {
    if (activeStep > 0) {
      setError("");
      setActiveStep((current) => current - 1);
    }
  };

  /* =========================================================
     REAL DRAG / SWIPE CAROUSEL
  ========================================================= */

  const handlePointerDown = (event) => {
    // Don't start carousel dragging when interacting with
    // buttons, inputs, labels, audio controls, etc.
    const target = event.target;

    if (
      target.closest(
        "button, input, label, a, audio"
      )
    ) {
      return;
    }

    dragStartX.current = event.clientX;
    dragCurrentX.current = event.clientX;

    setCarouselDragging(true);
    setDragOffset(0);

    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event) => {
    if (!carouselDragging) return;

    dragCurrentX.current = event.clientX;

    let offset =
      dragCurrentX.current - dragStartX.current;

    /*
      Prevent dragging too far beyond the first/last card.
      There is still a little resistance so the interface
      feels natural.
    */

    if (activeStep === 0 && offset > 0) {
      offset *= 0.25;
    }

    if (activeStep === 2 && offset < 0) {
      offset *= 0.25;
    }

    setDragOffset(offset);
  };

  const handlePointerUp = (event) => {
    if (!carouselDragging) return;

    const difference =
      dragCurrentX.current - dragStartX.current;

    const threshold = 100;

    if (Math.abs(difference) >= threshold) {
      if (difference < 0 && activeStep < 2) {
        goNext();
      } else if (
        difference > 0 &&
        activeStep > 0
      ) {
        goPrevious();
      }
    }

    setCarouselDragging(false);
    setDragOffset(0);

    try {
      event.currentTarget.releasePointerCapture(
        event.pointerId
      );
    } catch {
      // Pointer capture may already be released.
    }
  };

  const handlePointerCancel = (event) => {
    setCarouselDragging(false);
    setDragOffset(0);

    try {
      event.currentTarget.releasePointerCapture(
        event.pointerId
      );
    } catch {
      // Pointer capture may already be released.
    }
  };

  /* =========================================================
     BACKGROUND
  ========================================================= */

  const backgroundStyle = {
    backgroundImage: preview
      ? `url("${preview}")`
      : 'url("/vision-bg.png")',

    /*
      IMPORTANT:
      Cover makes the image fill the entire viewport.
      The uploaded image may be cropped slightly at the
      edges depending on its aspect ratio.
    */
    backgroundSize: "cover",
    backgroundPosition: "center center",
    backgroundRepeat: "no-repeat",

    /*
      Stop the uploaded image from drifting/zooming.
      The default background can still use the CSS animation.
    */
    animation: preview
      ? "none"
      : undefined,
  };

  /* =========================================================
     RENDER
  ========================================================= */

  return (
    <div className="vision-page">

      {/* =====================================================
          BACKGROUND
      ===================================================== */}

      <div
        className="background-image"
        style={backgroundStyle}
      />

      <div className="background-overlay" />
      <div className="background-vignette" />
      <div className="background-noise" />

      {/* Decorative elements */}
      <div className="glow glow-one" />
      <div className="glow glow-two" />

      <div className="floating-dot dot-one" />
      <div className="floating-dot dot-two" />
      <div className="floating-dot dot-three" />

      {/* =====================================================
          NAVBAR
      ===================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-symbol">
            <Sparkles size={17} />
          </div>

          <div>
            <h2>
              Vision<span>Caption</span>
            </h2>

            <small>
              SEE · UNDERSTAND · SPEAK
            </small>
          </div>

        </div>

        <nav>
          <button className="nav-active">
            Home
          </button>
        </nav>

        <div className="online-status">
          <span />
          AI ONLINE
        </div>

      </header>

      {/* =====================================================
          MAIN
      ===================================================== */}

      <main className="main">

        {/* ===================================================
            LEFT INTRO
        =================================================== */}

        <section className="intro">

          <div className="welcome-label">
            WELCOME
          </div>

          <h1 className="hero-title">
  <span className="hero-word word-1">Turn</span>{" "}
  <span className="hero-word word-2">images</span>
  <br />
  <span className="hero-word word-3">into</span>{" "}
  <span className="hero-word word-4 meaning-word">
    meaning.
  </span>
</h1>

          <p>
            Generate intelligent captions,
            translate them into your language,
            and hear your images come to life.
          </p>

          <div className="intro-line" />

        </section>

        {/* ===================================================
            DRAGGABLE CAROUSEL
        =================================================== */}

        <section
          className={`carousel-area ${
            carouselDragging
              ? "carousel-is-dragging"
              : ""
          }`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
        >

          <div
            className="carousel-track"
            style={{
              transform: `translateX(calc(-${
                activeStep * 33.333333
              }% + ${dragOffset}px))`,

              transition: carouselDragging
                ? "none"
                : "transform 0.65s cubic-bezier(0.22, 1, 0.36, 1)",
            }}
          >

            {/* =================================================
                CARD 01 — UPLOAD
            ================================================= */}

            <div className="carousel-slide">

              <div className="main-card">

                <div className="card-header">

                  <div>

                    <span className="step-label">
                      01
                    </span>

                    <h2>
                      Upload Image
                    </h2>

                    <p>
                      Start with an image you'd
                      like VisionCaption to understand.
                    </p>

                  </div>

                  <div className="card-icon">
                    <ImageIcon size={20} />
                  </div>

                </div>

                {/* Upload area */}

                <div
                  className={`upload-box ${
                    uploadDragging
                      ? "dragging"
                      : ""
                  }`}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setUploadDragging(true);
                  }}
                  onDragLeave={() => {
                    setUploadDragging(false);
                  }}
                  onDrop={(event) => {
                    event.preventDefault();

                    setUploadDragging(false);

                    handleImage(
                      event.dataTransfer.files[0]
                    );
                  }}
                >

                  {preview ? (

                    <div className="preview-image">

                      <img
                        src={preview}
                        alt="Uploaded"
                        draggable="false"
                      />

                      <label className="change-image">

                        <RotateCcw size={14} />

                        Change Image

                        <input
                          type="file"
                          accept="image/*"
                          hidden
                          onChange={(event) =>
                            handleImage(
                              event.target.files[0]
                            )
                          }
                        />

                      </label>

                    </div>

                  ) : (

                    <div className="upload-content">

                      <div className="upload-symbol">
                        <Upload size={27} />
                      </div>

                      <h3>
                        Drop your image here
                      </h3>

                      <p>
                        or click to browse
                        <br />
                        from your device
                      </p>

                      <label className="light-button">

                        <Upload size={15} />

                        Browse Image

                        <input
                          type="file"
                          accept="image/*"
                          hidden
                          onChange={(event) =>
                            handleImage(
                              event.target.files[0]
                            )
                          }
                        />

                      </label>

                      <small>
                        PNG · JPG · JPEG
                      </small>

                    </div>

                  )}

                </div>

                {/* Continue */}

                <button
                  className="continue-button"
                  disabled={!image}
                  onClick={goNext}
                >
                  Continue
                  <ArrowRight size={16} />
                </button>

              </div>

            </div>

            {/* =================================================
                CARD 02 — CONFIGURE
            ================================================= */}

            <div className="carousel-slide">

              <div className="main-card">

                <div className="card-header">

                  <div>

                    <span className="step-label">
                      02
                    </span>

                    <h2>
                      Configure
                    </h2>

                    <p>
                      Choose how VisionCaption
                      should understand your image.
                    </p>

                  </div>

                  <div className="card-icon">
                    <Cpu size={20} />
                  </div>

                </div>

                {/* MODEL */}

                <div className="config-section">

                  <div className="config-title">
                    <Cpu size={15} />
                    Captioning Model
                  </div>

                  <div className="model-options">

                    <button
                      className={
                        model === "attention"
                          ? "selected"
                          : ""
                      }
                      onClick={() =>
                        setModel("attention")
                      }
                    >

                      <span className="radio" />

                      <div>

                        <strong>
                          Attention
                        </strong>

                        <small>
                          Focuses on important
                          visual regions
                        </small>

                      </div>

                    </button>

                    <button
                      className={
                        model === "baseline"
                          ? "selected"
                          : ""
                      }
                      onClick={() =>
                        setModel("baseline")
                      }
                    >

                      <span className="radio" />

                      <div>

                        <strong>
                          Baseline
                        </strong>

                        <small>
                          Standard CNN-LSTM
                          captioning
                        </small>

                      </div>

                    </button>

                  </div>

                </div>

                {/* LANGUAGE */}

                <div className="config-section">

                  <div className="config-title">
                    <Languages size={15} />
                    Target Language
                  </div>

                  <div className="language-box">

                    <Languages size={17} />

                    <input
                      value={language}
                      onChange={(event) =>
                        setLanguage(
                          event.target.value
                        )
                      }
                      placeholder="Type a language..."
                    />

                  </div>

                  <small className="language-note">
                    Examples: English, Hindi,
                    Kannada, French, Spanish...
                  </small>

                </div>

                {/* ACTIONS */}

                <div className="config-actions">

                  <button
                    className="secondary-button"
                    onClick={goPrevious}
                  >
                    <ArrowLeft size={15} />
                    Back
                  </button>

                  <button
                    className="primary-button"
                    disabled={!image || loading}
                    onClick={generateCaption}
                  >

                    {loading ? (

                      <>
                        <span className="spinner" />
                        Processing...
                      </>

                    ) : (

                      <>
                        <Sparkles size={16} />
                        Generate Caption
                        <ArrowRight size={16} />
                      </>

                    )}

                  </button>

                </div>

                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}

              </div>

            </div>

            {/* =================================================
                CARD 03 — RESULT
            ================================================= */}

            <div className="carousel-slide">

              <div className="main-card result-card">

                {loading ? (

                  <div className="processing">

                    <div className="processing-icon">
                      <Sparkles size={27} />
                    </div>

                    <span className="step-label">
                      03
                    </span>

                    <h2>
                      Understanding your image
                    </h2>

                    <p>
                      VisionCaption is analyzing
                      the visual content...
                    </p>

                    <div className="loading-line">
                      <div />
                    </div>

                  </div>

                ) : result ? (

                  <>

                    <div className="card-header">

                      <div>

                        <span className="step-label">
                          03
                        </span>

                        <h2>
                          Your Result
                        </h2>

                        <p>
                          Here's what our AI sees.
                        </p>

                      </div>

                      <div className="card-icon">
                        <Sparkles size={20} />
                      </div>

                    </div>

                    {/* GENERATED CAPTION */}

                    <div className="output-box">

                      <span>
                        GENERATED CAPTION
                      </span>

                      <p>
                        “{result.caption}”
                      </p>

                    </div>

                    {/* TRANSLATION */}

                    <div className="output-box translated">

                      <span>
                        TRANSLATION ·{" "}
                        {result.language}
                      </span>

                      <p>
                        “
                        {result.translation ||
                          result.caption}
                        ”
                      </p>

                    </div>

                    {/* AUDIO */}

                    {result.audio_url && (

                      <div className="audio-box">

                        <div className="audio-header">

                          <div className="audio-icon">
                            <Volume2 size={17} />
                          </div>

                          <div>

                            <strong>
                              Listen to it
                            </strong>

                            <small>
                              AI generated speech
                            </small>

                          </div>

                        </div>

                        <audio
                          controls
                          src={`${API_BASE}${result.audio_url}`}
                        />

                        <a
                          href={`${API_BASE}${result.audio_url}`}
                          download
                          className="download-button"
                        >
                          <Download size={14} />
                          Download audio
                        </a>

                      </div>

                    )}

                    {/* RESULT ACTIONS */}

                    <div className="result-actions">

                      <button
                        className="secondary-button"
                        onClick={() =>
                          setActiveStep(1)
                        }
                      >
                        <ArrowLeft size={15} />
                        Adjust
                      </button>

                      <button
                        className="secondary-button"
                        onClick={resetAll}
                      >
                        <RotateCcw size={14} />
                        New Image
                      </button>

                    </div>

                  </>

                ) : (

                  <div className="empty-result">

                    <div className="processing-icon">
                      <Sparkles size={25} />
                    </div>

                    <span className="step-label">
                      03
                    </span>

                    <h2>
                      Your result will appear here
                    </h2>

                    <p>
                      Generate a caption to see
                      the AI's understanding.
                    </p>

                  </div>

                )}

              </div>

            </div>

          </div>

        </section>

      </main>

      {/* =====================================================
          FEATURE STRIP
      ===================================================== */}

      <section className="feature-strip">

        <div className="feature-item">

          <div className="feature-icon">
            <Cpu size={17} />
          </div>

          <div>

            <strong>
              Deep Learning
            </strong>

            <small>
              Visual understanding
            </small>

          </div>

        </div>

        <div className="feature-item">

          <div className="feature-icon">
            <Languages size={17} />
          </div>

          <div>

            <strong>
              Translation
            </strong>

            <small>
              Multiple languages
            </small>

          </div>

        </div>

        <div className="feature-item">

          <div className="feature-icon">
            <Volume2 size={17} />
          </div>

          <div>

            <strong>
              Text to Speech
            </strong>

            <small>
              Hear your caption
            </small>

          </div>

        </div>

        <div className="feature-quote">
          See the world.
          <br />
          <em>Put it into words.</em>
        </div>

      </section>

      {/* =====================================================
          FOOTER
      ===================================================== */}

      <footer>

        <span>
          © 2026 VisionCaption AI
        </span>

        <span>
          Vision · Language · Speech
        </span>

      </footer>

    </div>
  );
}

export default App;