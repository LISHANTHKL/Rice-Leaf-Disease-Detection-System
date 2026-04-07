from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from gtts import gTTS
from deep_translator import GoogleTranslator

# ---------------- APP SETUP ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
AUDIO_FOLDER = "static/audio"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("CNN_trained_model.h5")

_, IMG_H, IMG_W, IMG_C = model.input_shape
dummy = tf.zeros((1, IMG_H, IMG_W, IMG_C))
model(dummy)

# ---------------- LOAD CLASSES ----------------
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

class_names = {v: k for k, v in class_indices.items()}

# ---------------- DISEASE NAME TRANSLATIONS ----------------
disease_translation = {
    "Tungro": {"kn": "ಟಂಗ್ರೋ ರೋಗ", "hi": "तुंग्रो रोग"},
    "Blast": {"kn": "ಬ್ಲಾಸ್ಟ್ ರೋಗ", "hi": "ब्लास्ट रोग"},
    "Bacterialblight": {"kn": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಬ್ಲೈಟ್ ರೋಗ", "hi": "बैक्टीरियल ब्लाइट रोग"},
    "Brownspot": {"kn": "ಬ್ರೌನ್ ಸ್ಪಾಟ್ ರೋಗ", "hi": "ब्राउन स्पॉट रोग"}
}

# ---------------- GRAD-CAM ----------------
def generate_gradcam(model, img_array, img_path):

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    last_conv_index = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_index = i

    with tf.GradientTape() as tape:
        x = img_tensor
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == last_conv_index:
                conv_output = x
                tape.watch(conv_output)

        preds = x
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.7, heatmap_color, 0.5, 0)

    gradcam_path = os.path.join(GRADCAM_FOLDER, "gradcam_result.jpg")
    cv2.imwrite(gradcam_path, superimposed)

    return gradcam_path, heatmap_resized


# ---------------- SEVERITY ANALYSIS ----------------
def analyze_severity(heatmap):

    infected_ratio = np.sum(heatmap > 0.45) / heatmap.size

    if infected_ratio < 0.15:
        return "Mild", infected_ratio
    elif infected_ratio < 0.35:
        return "Moderate", infected_ratio
    else:
        return "Severe", infected_ratio


# ---------------- AI RECOMMENDATION ----------------
def ai_generate_solution(disease, severity):

    if disease == "Tungro":
        cause = "Tungro disease is caused by Rice Tungro Virus transmitted by green leafhopper insects."
        treatment = "Remove infected plants and control insects using recommended pesticides."

    elif disease == "Blast":
        cause = "Blast disease is a fungal infection caused by Magnaporthe oryzae."
        treatment = "Apply fungicides and maintain proper plant spacing."

    elif disease == "Bacterialblight":
        cause = "Bacterial blight is caused by Xanthomonas oryzae bacteria."
        treatment = "Use disease free seeds and avoid excess nitrogen fertilizer."

    elif disease == "Brownspot":
        cause = "Brown spot occurs due to fungal infection and poor soil nutrition."
        treatment = "Improve soil nutrients and apply fungicide if required."

    else:
        cause = "Disease affecting rice leaves."
        treatment = "Follow proper agronomic and chemical control methods."

    return {
        "cause": cause,
        "treatment": treatment,
        "severity_note": f"The infection severity is {severity}."
    }


# ---------------- AUTO TRANSLATION ----------------
def translate_text(text, lang):

    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except:
        return text


# ---------------- VOICE GENERATION ----------------
def generate_voice(solution, disease, severity):

    disease_kn = disease_translation[disease]["kn"]
    disease_hi = disease_translation[disease]["hi"]

    cause_kn = translate_text(solution["cause"], "kn")
    treat_kn = translate_text(solution["treatment"], "kn")

    cause_hi = translate_text(solution["cause"], "hi")
    treat_hi = translate_text(solution["treatment"], "hi")

    # English
    english_text = f"""
    Disease detected is {disease}.
    Severity level is {severity}.
    Cause: {solution['cause']}.
    Treatment: {solution['treatment']}.
    """

    # Kannada
    kannada_text = f"""
    ಗುರುತಿಸಲಾದ ರೋಗ {disease_kn}.
    ರೋಗದ ತೀವ್ರತೆ ಮಟ್ಟ {severity}.
    ಕಾರಣ: {cause_kn}.
    ಚಿಕಿತ್ಸೆ: {treat_kn}.
    """

    # Hindi
    hindi_text = f"""
    पहचानी गई बीमारी {disease_hi} है।
    बीमारी की गंभीरता {severity} है।
    कारण: {cause_hi}.
    उपचार: {treat_hi}.
    """

    en_path = os.path.join(AUDIO_FOLDER, "result_en.mp3")
    kn_path = os.path.join(AUDIO_FOLDER, "result_kn.mp3")
    hi_path = os.path.join(AUDIO_FOLDER, "result_hi.mp3")

    tts_en = gTTS(text=english_text, lang="en", slow=True)
    tts_kn = gTTS(text=kannada_text, lang="kn", slow=True)
    tts_hi = gTTS(text=hindi_text, lang="hi", slow=True)

    tts_en.save(en_path)
    tts_kn.save(kn_path)
    tts_hi.save(hi_path)

    return en_path, hi_path, kn_path


# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        preds = model.predict(img)
        confidence = float(np.max(preds) * 100)

        if confidence < 60:
            return render_template(
                "index.html",
                error="❌ Invalid image! Please upload a clear rice leaf image."
            )

        predicted_class = class_names[int(np.argmax(preds))]

        # Gradcam
        gradcam_path, heatmap = generate_gradcam(model, img, filepath)

        # Severity
        severity, ratio = analyze_severity(heatmap)

        # Recommendation
        solution = ai_generate_solution(predicted_class, severity)

        # Voice
        voice_en, voice_hi, voice_kn = generate_voice(
            solution, predicted_class, severity
        )

        return render_template(
            "result.html",
            image_path=filepath,
            gradcam_path=gradcam_path,
            prediction=predicted_class,
            confidence=round(confidence, 2),
            solution=solution,
            severity=severity,
            voice_en=voice_en,
            voice_hi=voice_hi,
            voice_kn=voice_kn
        )

    return render_template("index.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)


