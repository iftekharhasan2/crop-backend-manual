from flask import Flask, request, jsonify
import numpy as np
import io
from PIL import Image
from tensorflow import keras
import requests
from flask_cors import CORS
from datetime import date

app = Flask(__name__)
CORS(app)





def day01():
    a1 = "সূর্যের আলোয় ৪ ঘণ্টা বীজ শুকিয়ে নিন।"
    a2 = "সকাল"
    a3 = "৭টা – ১২টা"
    b1 = "স্বাভাবিক তাপমাত্রায় বীজকে ঠান্ডা করে নিন।"
    b2 = "দুপুর"
    b3 = "১২টা – ৪টা"
    c1 = "পানিতে ৬ ঘণ্টা ভিজিয়ে রাখুন।"
    c2 = "বিকাল – রাত"
    c3 = "৪টা – ১০টা"
    return a1, a2, a3, b1, b2, b3, c1, c2, c3

def day02():
    a1 = "প্রতি কেজি বীজের সাথে ১০ গ্রাম ট্রাইকোডার্মা মিশিয়ে নিন।"
    a2 = "সকাল"
    a3 = "৬টা – ৭টা"
    b1 = "বীজকে শুকিয়ে নিন (সরাসরি সূর্যের তাপ দেয়া যাবেনা)"
    b2 = "সকাল"
    b3 = "৭টা – ১০টা"
    c1 = "বীজকে ২গ্রাম/লিটার জিবেরেলিক এসিড (GA3 10%) মিশ্রনে ১০ মিনিট ডুবিয়ে রেখে বাতাসে শুকিয়ে নিন।"
    c2 = "সকাল"
    c3 = "১০টা – ১১টা"
    d1 = "একটি হালকা আর্দ্র কাপড়ে বীজ বেঁধে পুত্তলি বানিয়ে ২১ ঘণ্টা শুষ্ক জায়গায় রাখুন।"
    d2 = "সকাল"
    d3 = "১১টা – ৮টা"
    return a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3

def day03():
    a1 = "বীজ বপন করুন এবং ছায়াযুক্ত স্থানে রাখুন।"
    a2 = "সকাল"
    a3 = "৬টা – ৮টা"
    b1 = "বীজ বপনের সময় অঙ্কুরোদ্গম না"
    b2 = "সকাল"
    b3 = "৬টা – ৮টা"
    sp_statement01 = (
        "বীজ বপন পদ্ধতিঃ "
        "বীজকে প্রথমে অঙ্কুরোদ্গম করে নিতে হবে। "
        "শীতকালে খুব ঠাণ্ডা থাকলে বীজ ১২ ঘন্টা পানিতে ভিজিয়ে রেখে গোবরের মাদার ভেতরে কিংবা মাটির পাত্রে রক্ষিত বালির ভেতরে রেখে দিলে ২-৩ দিনের মধ্যে বীজ অঙ্কুরিত হয়। "
        "জমি থেকে এক মুষ্ঠি পরিমান মাটি তুলে এক মুষ্ঠি পরিমান ট্রাইকোডার্মা মিশ্রিত জৈব সার মিশিয়ে নিতে হবে। "
        "বীজকে ২ সেমি (আঙ্গুলের ২ করা) গভীরে বপন করতে হবে, জৈব সার মিশ্রিত মাটি আলতো করে দিয়ে দিতে হবে। "
        "চারা গজিয়ে গেলে প্রতিদিন সকালে ঝাঝরি দিয়ে চারার গোড়ার অঞ্চল হালকা করে ভিজিয়ে দিতে হবে (শীতকালে এবং জমি আর্দ্র থাকলে দরকার নেই)।"
    )
    sp_statement02 = (
        "সতর্কীকরণঃ "
        "চারার সুস্থ বৃদ্ধির জন্য শিডিউল স্প্রে জাতীয় পুষ্টি উপাদান স্প্রে করে দিতে হবে। "
        "চারা গজানোর পর জমিতে বেডের উচ্চতার ৩ ভাগের একভাগ পানি দিতে হবে, তবে খেয়াল রাখতে যেন চারার গোড়ায় পানি না জমে থাকে। "
        "বৃষ্টির সময় অথবা শীতকালে জমি আর্দ্র থাকলে সেচের দরকার নেই। "
        "শীতকালে বেড আর্দ্র থাকলে ৩ ভাগের একভাগ পানি দেয়ার দরকার নেই, তবে গ্রীষ্মকালে ১৫ দিন পর পর সেচ দিতে হবে এবং বেডের ৩ ভাগের এক ভাগ পানি নিশ্চিত করতে হবে। "
        "চারার বয়স ১০-১৫ দিন হয়ে গেলে তখন থেকে ঢালাও ভাবে সেচ দিতে হবে, তবে শীতকালে দেয়ার প্রয়োজন নেই।"
    )
    return a1, a2, a3, b1, b2, b3, (sp_statement01, sp_statement02)

def day_seed02():
    a1 = "দমন ব্যবস্থাপনা ---> কল্যাণ ৩ গ্রাম/লিটার + নেয়ামত ০.৫গ্রাম/লিটার"
    a2 = "সকাল"
    a3 = "৭-৮টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন।"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    sp_statement01 = "if u already done it, then u dont have to do anything"
    return a1, a2, a3, b1, b2, b3, (sp_statement01,)

def day_seed05():
    a1 = "দমন ব্যবস্থাপনা ---> ক্রপ মাস্টার ৫ গ্রাম/লিটার + ম্যাট্রিক্সিন ১ মিলি/লিটার"
    a2 = "সকাল"
    a3 = "৭-৮টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন।"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    return a1, a2, a3, b1, b2, b3

def day_seed07():
    a1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন। এর জায়গা হাল্কা ফাকা করে আগাছা তুলে ফেলতে হবে যেন গাছের গোড়ার ক্ষতি না হয়। "
    a2 = "সকাল"
    a3 = "৯-১১টা"
    return a1, a2, a3

def day_seed10():
    a1 = "দমন ব্যবস্থাপনা ---> কল্যান ৩ গ্রাম/লিটার + প্রোফাইটা ১ মিলি/লিটার + ইয়োকা ১ মিলি/লিটার"
    a2 = "সকাল"
    a3 = "৭-৮টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন।"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    return a1, a2, a3, b1, b2, b3

def day_seed12():
    a1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন। চারার গোড়ার আগাছা তোলাঃ চারার গোড়ার এক হাত সমপরিমানে চতুর্দিক থেকে আগাছা তুলে ফেলুন। মালচিং এর জায়গা হাল্কা ফাকা করে আগাছা তুলে ফেলতে হবে যেন গাছের গোড়ার ক্ষতি না হয়। ড্রেনের আগাছা তোলাঃ এক্ষেত্রে ড্রেন থেকে মূলসহ আগাছা উপড়ে ফেলতে হবে।"
    a2 = "সকাল"
    a3 = "৯-১১টা"
    return a1, a2, a3

def day_seed15():
    a1 = "দমন ব্যবস্থাপনা ---> গ্রিন প্লাস ১মিলি/লিটার + চিলেটেট জিঙ্ক ০.৫ গ্রাম/লিটার"
    a2 = "সকাল"
    a3 = "৭-৮টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    return a1, a2, a3, b1, b2, b3

def day_seed17():
    a1 = "উপরি সার প্রয়োগ ----> ১ম কিস্তি উপরি সার প্রয়োগঃ DAP – ২০০ গ্রাম/শতক + TSP – ১৫০গ্রাম/শতক"
    a2 = "সকাল"
    a3 = "৮-৯.৩০টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    return a1, a2, a3, b1, b2, b3

def day_seed19():
    a1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন। চারার গোড়ার আগাছা তোলাঃ চারার গোড়ার এক হাত সমপরিমানে চতুর্দিক থেকে আগাছা তুলে ফেলুন। মালচিং এর জায়গা হাল্কা ফাকা করে আগাছা তুলে ফেলতে হবে যেন গাছের গোড়ার ক্ষতি না হয়। ড্রেনের আগাছা তোলাঃ এক্ষেত্রে ড্রেন থেকে মূলসহ আগাছা উপড়ে ফেলতে হবে।"
    a2 = "সকাল"
    a3 = "৯-১১টা"
    return a1, a2, a3

def day_seed20():
    a1 = "দমন ব্যবস্থাপনা ---> কপাল ২ গ্রাম/লিটার + নেয়ামত ০.৫ গ্রাম/লিটার"
    a2 = "সকাল"
    a3 = "৭-৮টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    return a1, a2, a3, b1, b2, b3

def day_seed25():
    a1 = "দমন ব্যবস্থাপনা ---> গ্রিন প্লাস ১মিলি/লিটার + চিলেটেট জিঙ্ক ০.৫ গ্রাম/লিটার"
    a2 = "সকাল"
    a3 = "৭-৮টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    return a1, a2, a3, b1, b2, b3

def day_seed26():
    a1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন। চারার গোড়ার আগাছা তোলাঃ চারার গোড়ার এক হাত সমপরিমানে চতুর্দিক থেকে আগাছা তুলে ফেলুন। মালচিং এর জায়গা হাল্কা ফাকা করে আগাছা তুলে ফেলতে হবে যেন গাছের গোড়ার ক্ষতি না হয়। ড্রেনের আগাছা তোলাঃ এক্ষেত্রে ড্রেন থেকে মূলসহ আগাছা উপড়ে ফেলতে হবে।"
    a2 = "সকাল"
    a3 = "৯-১১টা"
    return a1, a2, a3

def day_seed30():
    a1 = "দমন ব্যবস্থাপনা ---> সম্পদ ১.৫ গ্রাম/লিটার + ইয়োকো ১ মিলি/লিটার + স্টারথেন ১ গ্রাম/লিটার"
    a2 = "সকাল"
    a3 = "৭-৮টা"
    b1 = "পরিচর্যা ---> কোন চারা রোগাক্রান্ত হলে তা দ্রুত তুলে ফেলুন। জমিতে রসের অভাব দেখা দিলে সেচ দিন"
    b2 = "সকাল"
    b3 = "৯-১১টা"
    return a1, a2, a3, b1, b2, b3


def center(day, crop="cucumber", city="Dhaka", phase="জমি প্রস্তুতকালীন সময়কাল"):
    temp, humidity, weather_desc, wind_speed = weather_report(city)
    if temp is None:
        return "Weather data unavailable. Try again later.", None, None, None, None
    if crop.lower() == "cucumber":
        return cucumber(day, temp, humidity, weather_desc, wind_speed, phase), temp, humidity, weather_desc, wind_speed
    return f"No logic defined for crop: {crop}", None, None, None, None


def cucumber(day, temp, humidity, weather_desc, wind_speed, phase):
    if weather_desc:
        if phase == "জমি প্রস্তুতকালীন সময়কাল":
            if day == 1:
                return day01()
            elif day == 2:
                return day02()
            elif day == 3:
                return day03()
            else:
                return "এই দিনের জন্য কোনো নির্দেশনা পাওয়া যায়নি।"
        elif phase == "সংবেদনশীল সময়কাল":
            if day in [1, 2, 3, 4]:
                return day_seed02()
            elif day in [5, 6]:
                return day_seed05()
            elif day in [7, 8, 9]:
                return day_seed07()
            elif day in [10, 11]:
                return day_seed10()
            elif day in [12, 13, 14]:
                return day_seed12()
            elif day in [15, 16]:
                return day_seed15()
            elif day in [17, 18]:
                return day_seed17()
            elif day == 19:
                return day_seed19()
            elif day in [20, 21, 22, 23, 24]:
                return day_seed20()
            elif day == 25:
                return day_seed25()
            elif day in [26, 27, 28, 29]:
                return day_seed26()
            elif day == 30:
                return day_seed30()
            else:
                return "এই দিনের জন্য কোনো নির্দেশনা পাওয়া যায়নি।"
        else:
            return "পর্যায়ের তথ্য সঠিক নয়।"
    return "এই আবহাওয়ায় চাষের জন্য উপযুক্ত পরিকল্পনা পাওয়া যায়নি।"




# Load ML Model
model = keras.models.load_model('tomato_disease.h5')

class_labels = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

disease_prevention = {
    "Tomato_Bacterial_spot": [
        "Prevent bacterial spot by using disease-free seeds.",
        "Implement crop rotation to reduce the disease's prevalence.",
        "Apply copper-based fungicides to control the disease."
    ],
    "Tomato_Early_blight": [
        "Prevent early blight by practicing good garden hygiene.",
        "Ensure proper watering to avoid splashing soil onto the leaves.",
        "Apply fungicides as needed to control the disease."
    ],
    "Tomato_Late_blight": [
        "Prevent late blight by providing good air circulation in your garden or greenhouse.",
        "Avoid overhead watering, as wet leaves can encourage the disease.",
        "Apply fungicides when necessary to manage the disease."
    ],
    "Tomato_Leaf_Mold": [
        "Prevent leaf mold by ensuring good air circulation and spacing between plants.",
        "Avoid wetting the leaves when watering, and water the soil instead.",
        "Apply fungicides if the disease is present and worsening."
    ],
    "Tomato_Septoria_leaf_spot": [
        "Prevent Septoria leaf spot by maintaining good garden hygiene.",
        "Avoid overhead watering to keep the leaves dry.",
        "Apply fungicides if the disease becomes a problem."
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "Inspect plants regularly for signs of infestation.",
        "Increase humidity to discourage mites.",
        "Use insecticidal soap or neem oil to control mites."
    ],
    "Tomato__Target_Spot": [
        "Ensure good air circulation and avoid overcrowding of plants.",
        "Water at the base to keep leaves dry.",
        "Apply fungicides as needed."
    ],
    "Tomato__Tomato_YellowLeaf__Curl_Virus": [
        "Use virus-free tomato plants.",
        "Control whiteflies with insecticides.",
        "Remove and destroy infected plants."
    ],
    "Tomato__Tomato_mosaic_virus": [
        "Use virus-free seeds and disease-resistant varieties.",
        "Control aphids with insecticides.",
        "Remove and destroy infected plants."
    ],
    "Tomato_healthy": [
        "Continue monitoring for pests and diseases.",
        "Practice proper watering and fertilization."
    ]
}

def read_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    img = np.array(img)
    return img

@app.route("/api/detect", methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = read_image(file.read())
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        pred_index = np.argmax(prediction)
        confidence = float(np.max(prediction[0]))
        predicted_class = class_labels[pred_index]
        prevention = disease_prevention.get(predicted_class, ['Prevention info not available.'])
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'prevention_measures': prevention
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def weather_report(city):
    api_key = "1bd0be556664fb2dcaa474e56927746d"
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        return (
            data["main"]["temp"],
            data["main"]["humidity"],
            data["weather"][0]["main"],
            data["wind"]["speed"]
        )
    except requests.RequestException:
        return None, None, "Unknown", None

@app.route("/api/weather", methods=["POST"])
def api_weather():
    data = request.get_json()
    city = data.get("city", "Dhaka")
    temp, humidity, weather_desc, wind_speed = weather_report(city)
    if temp is None:
        return jsonify({"error": "Could not fetch weather. Check your city name."}), 400
    return jsonify({
        "Temperature": temp,
        "Humidity": humidity,
        "Weather": weather_desc,
        "Wind Speed": wind_speed
    }), 200

@app.route("/api/crop_manuals", methods=["POST"])
def api_crop_manuals():
    data = request.get_json()
    start_date = data.get("start_date", date.today().strftime("%Y-%m-%d"))
    city = data.get("city", "Dhaka")
    phase = data.get("phase", "জমি প্রস্তুতকালীন সময়কাল")
    day = int(data.get("day", 1))
    chara = data.get("chara", "No")
    dynamic_choice = data.get("dynamic_choice", "Yes")

    if dynamic_choice == "No":
        return jsonify({"message": "Redirect to manual"}), 200
    if phase == "সংবেদনশীল সময়কাল" and chara == "No":
        return jsonify({"error": "চারা গজানোর জন্য অপেক্ষা করুন।"}), 400

    result, temp, humidity, weather_desc, wind_speed = center(day, "cucumber", city, phase)
    if temp is None:
        return jsonify({"error": "Weather data unavailable. Try again later."}), 500

    step_items = [item for item in result if isinstance(item, str)]
    tuple_items = [item for item in result if isinstance(item, tuple)]
    steps = len(step_items) // 3
    step_numbers = [str(i + 1) for i in range(steps)]
    descriptions = [step_items[i * 3] for i in range(steps)]
    times_of_day = [step_items[i * 3 + 1] for i in range(steps)]
    time_ranges = [step_items[i * 3 + 2] for i in range(steps)]

    steps_data = [
        {"Step": num, "Task Description": desc, "Time of Day": tod, "Time Range": tr}
        for num, desc, tod, tr in zip(step_numbers, descriptions, times_of_day, time_ranges)
    ]

    return jsonify({
        "data": steps_data,
        "extra_notes": tuple_items,
        "weather": {
            "temperature": temp,
            "humidity": humidity,
            "description": weather_desc,
            "wind_speed": wind_speed
        },
        "warning": "It is not a good weather for farming. Please try again Tomorrow." if weather_desc in ["Rain", "Haze"] else None
    }), 200

# Example control functions (day01, day02, etc.) and center() would be defined here as in original file.
# For brevity, they are assumed to be already defined.

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

