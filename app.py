import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
import tempfile
import cv2
import numpy as np
from google import genai
from orders_database import orders_database

# =========================
# LOAD API KEY
# =========================
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

IMAGE_MODEL = "gemini-2.5-flash"

# =========================
# HELPER FUNCTIONS
# =========================

def notify_customer_service(customer_name, order_id, complaint_item, damage_result, customer_data):
    """
    Notifikasi ke customer service untuk komplain yang memerlukan verifikasi lebih lanjut.
    Menyimpan data ke file log untuk customer service team.
    """
    import json
    from datetime import datetime

    notification = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": customer_name,
        "order_id": order_id,
        "complaint_item": complaint_item,
        "damage_result": damage_result,
        "days_after_delivery": customer_data.get("days_after_delivery", 0),
        "previous_complaints": customer_data.get("previous_complaints", 0),
        "status": "REQUIRES_VERIFICATION",
        "notes": "Komplain terlambat. Video unboxing tidak tersedia. Memerlukan verifikasi manual dari customer service."
    }

    # Simpan ke file log
    log_file = "customer_service_notifications.json"
    notifications = []

    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                notifications = json.load(f)
        except:
            notifications = []

    notifications.append(notification)

    with open(log_file, "w") as f:
        json.dump(notifications, f, indent=2)

    return True

def notify_store_compensation(customer_name, order_id, complaint_item, damage_result, fraud_level, complaint_status, compensation_choice=None):
    """
    Notifikasi ke toko tentang keputusan kompensasi dan pilihan customer.
    Menyimpan data ke file log untuk toko/warehouse team.
    """
    import json
    from datetime import datetime

    notification = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": customer_name,
        "order_id": order_id,
        "complaint_item": complaint_item,
        "damage_result": damage_result,
        "fraud_level": fraud_level,
        "complaint_status": complaint_status,
        "compensation_choice": compensation_choice,
        "action_required": True if complaint_status == "APPROVED" else False
    }

    # Simpan ke file log
    log_file = "store_compensation_notifications.json"
    notifications = []

    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                notifications = json.load(f)
        except:
            notifications = []

    notifications.append(notification)

    with open(log_file, "w") as f:
        json.dump(notifications, f, indent=2)

    return True

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_app_state():
    reset_keys = [
        "customer_name", "order_id", "complaint_item", "complaint_reason",
        "image_analyzed", "damage_result", "late_complaint", "fraud_score",
        "fraud_level", "complaint_status", "compensation_result",
        "verification_note", "show_video_uploader", "video_checked",
        "video_damage_result", "video_frames", "contacted_customer_service",
        "compensation_choice", "compensation_notified", "finish_and_reset",
        "final_message", "show_reset_button"
    ]
    for key in reset_keys:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.uploader_key += 1
    st.rerun()

def analyze_image_with_gemini(image_obj):
    prompt = """
Kamu adalah AI Fraud Detection untuk e-commerce barang pecah belah.

Tugas:
1. Analisa tingkat kerusakan barang dari gambar.
2. Tentukan salah satu:
    - Tingkat Kerusakan Barang: RINGAN
    - Tingkat Kerusakan Barang: SEDANG
    - Tingkat Kerusakan Barang: BERAT

Rules:
- Jawaban HARUS Salah 1 dari:
    - Tingkat Kerusakan Barang: RINGAN
    - Tingkat Kerusakan Barang: SEDANG
    - Tingkat Kerusakan Barang: BERAT
- Jangan memberi penjelasan tambahan.
"""
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=[prompt, image_obj]
    )
    return response.text.strip()


def extract_video_frames(video_file, max_frames=3):
    video_file.seek(0)
    suffix = os.path.splitext(video_file.name)[1] if hasattr(video_file, "name") else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.read())
        temp_name = tmp.name

    frames = []
    cap = cv2.VideoCapture(temp_name)
    if not cap.isOpened():
        cap.release()
        os.unlink(temp_name)
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        os.unlink(temp_name)
        return frames

    interval = max(1, total_frames // max_frames)
    for idx in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        if len(frames) >= max_frames:
            break

    cap.release()
    os.unlink(temp_name)
    return frames

# =========================
# TITLE & SIDEBAR
# =========================
st.title("🤖 Layanan Customer Care AI")

with st.sidebar:
    st.write("### ⚙️ Pengaturan")
    if st.button("🧹 Bersihkan Obrolan", use_container_width=True):
        reset_app_state()

with st.chat_message("assistant"):
    st.write(
        "Halo Kak 👋\n"
        "Selamat datang di Layanan Customer Care kami. Kami sangat menyesal mendengar Kakak mengalami kendala pada pesanan yang diterima.\n\n"
        "Agar kami dapat segera memberikan solusi terbaik, mohon bantuannya untuk mengisi detail pesanan dan mengunggah foto bukti kerusakannya di bawah ini ya Kak 🙏😊"
    )

# =========================
# STREAMLIT STATE
# =========================
if "image_analyzed" not in st.session_state:
    st.session_state.customer_name = ""
    st.session_state.order_id = ""
    st.session_state.complaint_item = ""
    st.session_state.complaint_reason = ""
    st.session_state.image_file = None
    st.session_state.video_file = None
    st.session_state.image_analyzed = False
    st.session_state.damage_result = ""
    st.session_state.late_complaint = False
    st.session_state.fraud_score = 0
    st.session_state.fraud_level = "LOW"
    st.session_state.complaint_status = ""
    st.session_state.compensation_result = ""
    st.session_state.verification_note = ""
    st.session_state.show_video_uploader = False
    st.session_state.video_checked = False
    st.session_state.video_damage_result = ""
    st.session_state.video_frames = []
    st.session_state.contacted_customer_service = False
    st.session_state.compensation_choice = None
    st.session_state.compensation_notified = False
    st.session_state.finish_and_reset = False
    st.session_state.final_message = ""
    st.session_state.show_reset_button = False

# =========================
# CUSTOMER COMPLAINT FORM
# =========================
with st.chat_message("user"):
    st.write("**📝 Formulir Pengajuan Komplain**")
    customer_name = st.text_input("Nama Lengkap Kakak:", key="customer_name")
    order_id = st.text_input("Kode Pesanan:", key="order_id")
    complaint_item = st.text_input("Barang yang Berkendala:", key="complaint_item")
    complaint_reason = st.text_area("Boleh ceritakan detail kendala yang Kakak alami?", key="complaint_reason")

    image_file = st.file_uploader(
        "Mohon unggah foto barang yang berkendala di sini 👇",
        type=["jpg", "jpeg", "png"],
        key=f"image_file_{st.session_state.uploader_key}"
    )

    image = None
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Foto barang yang diunggah", use_container_width=True)

    analyze_image_button = st.button("Kirim Data Komplain Saya 📥")

customer_data = None
if order_id:
    if order_id in orders_database:
        customer_data = orders_database[order_id]
    else:
        with st.chat_message("assistant"):
            st.error("Mohon maaf Kak, Kode Pesanan tidak dapat kami temukan di sistem kami. Mohon periksa kembali ya 🙏")
        st.stop()

if customer_data and customer_data["days_after_delivery"] > 3 and not st.session_state.image_analyzed:
    with st.chat_message("assistant"):
        st.info(
            "Foto yang Kakak unggah akan kami analisis terlebih dahulu ya. Namun, karena pesanan ini sudah lewat dari batas waktu garansi standar, kami mungkin akan meminta Kakak untuk mengunggah video unboxing paketnya nanti untuk proses verifikasi lanjutan 🙏"
        )

# =========================
# PROCESS IMAGE
# =========================
if analyze_image_button:
    if not order_id:
        st.error("Mohon isi Kode Pesanan terlebih dahulu ya Kak.")
    elif image is None:
        st.error("Mohon unggah foto barang yang berkendala agar dapat kami bantu proses.")
    else:
        with st.spinner("Mohon ditunggu sebentar ya Kak, kami sedang memeriksa detail dan foto yang Kakak kirimkan... 🙏😊"):
            st.session_state.late_complaint = customer_data["days_after_delivery"] > 3
            damage_result = analyze_image_with_gemini(image)
            st.session_state.damage_result = damage_result
            st.session_state.image_analyzed = True
            st.session_state.video_checked = False

            fraud_score = 0
            if customer_data["purchase_item"] != complaint_item:
                fraud_score += 50
            if customer_data["days_after_delivery"] > 3:
                fraud_score += 30
            if customer_data["previous_complaints"] >= 3:
                fraud_score += 20
            if "Tingkat Kerusakan Barang: RINGAN" in damage_result:
                fraud_score += 40
            elif "Tingkat Kerusakan Barang: SEDANG" in damage_result:
                fraud_score += 15

            if fraud_score >= 70:
                fraud_level = "HIGH"
            elif fraud_score >= 40:
                fraud_level = "MEDIUM"
            else:
                fraud_level = "LOW"

            st.session_state.fraud_score = fraud_score
            st.session_state.fraud_level = fraud_level

            if st.session_state.late_complaint or fraud_level in ["MEDIUM", "HIGH"]:
                st.session_state.show_video_uploader = True
            else:
                st.session_state.show_video_uploader = False

# =========================
# DISPLAY RESULTS
# =========================
if st.session_state.image_analyzed:
    with st.chat_message("assistant"):
        st.success(f"Berdasarkan analisis foto, sistem kami mendeteksi: **{st.session_state.damage_result}**")

        if customer_data:
            late_complaint = customer_data["days_after_delivery"] > 3
            if late_complaint:
                st.info("Karena pesanan ini sudah diterima cukup lama, kami butuh bantuan ekstra nih Kak. Boleh minta tolong unggah video unboxing-nya untuk proses verifikasi lebih lanjut? 🙏")
            elif st.session_state.fraud_level in ["MEDIUM", "HIGH"]:
                st.warning("🚨 Agar kami bisa memproses komplain Kakak dengan adil, mohon kesediaannya untuk mengunggah video unboxing paketnya ya Kak 🙏")

    if st.session_state.show_video_uploader:
        with st.chat_message("user"):
            st.write("**🎥 Unggah Video Unboxing**")
            video_file = st.file_uploader(
                "Silakan unggah video unboxing Kakak di sini 👇",
                type=["mp4", "mov", "avi", "webm"],
                key=f"video_file_{st.session_state.uploader_key}"
            )
            if video_file is not None:
                st.video(video_file)
                st.write("Video sudah siap diperiksa.")
                analyze_video_button = st.button("Kirim Video Unboxing Saya 🎥")

            if 'analyze_video_button' in locals() and analyze_video_button and video_file is not None:
                with st.spinner("Mohon ditunggu sebentar ya Kak, kami sedang memeriksa video unboxing Kakak... 🙏😊"):
                    frames = extract_video_frames(video_file, max_frames=3)
                    frame_results = []
                    for frame in frames:
                        frame_result = analyze_image_with_gemini(frame)
                        if frame_result in ["Tingkat Kerusakan Barang: RINGAN", "Tingkat Kerusakan Barang: SEDANG", "Tingkat Kerusakan Barang: BERAT"]:
                            frame_results.append(frame_result)
                    if frame_results:
                        severity_order = {"Tingkat Kerusakan Barang: RINGAN": 0, "Tingkat Kerusakan Barang: SEDANG": 1, "Tingkat Kerusakan Barang: BERAT": 2}
                        video_result = max(frame_results, key=lambda x: severity_order.get(x, 0))
                        st.session_state.video_damage_result = video_result
                        st.session_state.video_checked = True
                        st.success(f"✅ Video unboxing berhasil diverifikasi. Hasil pengecekan kami: {video_result}")
                    else:
                        st.warning("Mohon maaf Kak, kami kesulitan menentukan tingkat kerusakan dari video tersebut.")

    if not st.session_state.show_video_uploader:
        with st.chat_message("assistant"):
            st.divider()
            st.write("**🤖 Keputusan Tim Customer Care**")
            damage_result = st.session_state.damage_result
            fraud_level = st.session_state.fraud_level

            if fraud_level == "HIGH":
                complaint_status = "REJECTED"
                compensation_result = "❌ Mohon maaf sebesar-besarnya Kak, untuk saat ini komplain belum dapat kami setujui karena sistem kami mendeteksi perlunya pengecekan manual secara lebih mendetail.\nNamun jangan khawatir, kami akan segera menghubungkan Kakak dengan tim Customer Service kami untuk bantuan lebih lanjut.\nSekali lagi, kami mohon maaf atas ketidaknyamanan ini 🙏😊"
            elif fraud_level == "MEDIUM":
                complaint_status = "NEED VERIFICATION"
                if "Tingkat Kerusakan Barang: RINGAN" in damage_result:
                    compensation_result = "Kami sangat memahami kekecewaan Kakak 🙏\nSebagai permohonan maaf kami, kami ingin memberikan voucher diskon belanja sebesar 10% untuk Kakak.\nVoucher ini akan segera kami proses dalam 2-3 hari kerja ya Kak. Terima kasih banyak atas kesabaran dan pengertiannya 🙏😊"
                else:
                    compensation_result = "📋 Untuk memastikan solusi terbaik, kami memerlukan sedikit bantuan dari Kakak. Mohon berkenan mengirimkan video unboxing paket tersebut agar kami dapat memproses komplain ini lebih lanjut 🙏"
            else:
                complaint_status = "APPROVED"
                if "Tingkat Kerusakan Barang: RINGAN" in damage_result:
                    compensation_result = "Kami sangat memahami kekecewaan Kakak 🙏\nSebagai bentuk permohonan maaf, kami ingin memberikan voucher diskon belanja sebesar 20% untuk Kakak.\nVoucher ini akan segera kami proses dalam 2-3 hari kerja. Terima kasih banyak atas kesabaran Kakak 🙏😊"
                elif "Tingkat Kerusakan Barang: SEDANG" in damage_result:
                    compensation_result = "Kami sangat menyesal atas kejadian ini Kak 🙏\nSebagai bentuk tanggung jawab kami, kami akan mengembalikan 50% dari dana Kakak (partial refund).\nProses ini akan memakan waktu 2-3 hari kerja ya Kak. Terima kasih atas pengertiannya 🙏😊"
                elif "Tingkat Kerusakan Barang: BERAT" in damage_result:
                    compensation_result = "Kami sangat memohon maaf atas kerusakan parah pada barang Kakak 🙏 Karena hal ini, silakan pilih opsi kompensasi yang paling nyaman untuk Kakak di bawah ini 👇"

            customer_response = f"""
Halo Kak {customer_name},

Berdasarkan pengecekan sistem kami, kami mendeteksi:
**{damage_result}**

{compensation_result}
"""
            st.write(customer_response)

            if complaint_status == "APPROVED" and "Tingkat Kerusakan Barang: BERAT" in damage_result:
                st.divider()
                st.write("**💳 Pilih Opsi Kompensasi Kakak**")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("💰 Full Refund (Pengembalian Dana Penuh)", key="refund_btn_no_video", use_container_width=True):
                        notify_store_compensation(customer_name, order_id, complaint_item, damage_result, fraud_level, "APPROVED", "Full Refund")
                        st.session_state.compensation_choice = "Full Refund"
                        st.session_state.compensation_notified = True
                        st.session_state.final_message = (
                            "✅ Pilihan kompensasi Kakak telah kami sampaikan ke pihak gudang. "
                            "Kami akan memproses Full Refund Kakak dalam 2-3 hari kerja. "
                            "Terima kasih banyak atas kesabaran dan kebaikannya Kak 🙏😊"
                        )
                        st.session_state.show_reset_button = True

                with col2:
                    if st.button("📦 Pengiriman Ulang Barang Baru", key="reshipping_btn_no_video", use_container_width=True):
                        notify_store_compensation(customer_name, order_id, complaint_item, damage_result, fraud_level, "APPROVED", "Pengiriman Ulang Barang Baru")
                        st.session_state.compensation_choice = "Pengiriman Ulang Barang Baru"
                        st.session_state.compensation_notified = True
                        st.session_state.final_message = (
                            "✅ Pilihan kompensasi Kakak telah kami sampaikan ke pihak gudang. "
                            "Kami akan memproses pengiriman ulang barang baru untuk Kakak dalam 2-3 hari kerja. "
                            "Terima kasih banyak atas kesabaran dan kebaikannya Kak 🙏😊"
                        )
                        st.session_state.show_reset_button = True

                if st.session_state.compensation_notified:
                    st.divider()
                    st.success(f"✅ Opsi yang Kakak pilih: **{st.session_state.compensation_choice}**")
                    st.info(st.session_state.final_message)

                if st.session_state.show_reset_button:
                    st.divider()
                    st.write("\n")
                    if st.button("🔄 Kembali ke Halaman Utama", key="reset_final_btn_no_video"):
                        reset_app_state()

    elif st.session_state.show_video_uploader and st.session_state.video_checked:
        with st.chat_message("assistant"):
            st.divider()
            st.write("**🤖 Keputusan Tim Customer Care (Setelah Cek Video)**")

            damage_result = st.session_state.video_damage_result
            fraud_score = 0
            if customer_data["purchase_item"] != complaint_item:
                fraud_score += 50
            if customer_data["days_after_delivery"] > 3:
                fraud_score += 10 
            if customer_data["previous_complaints"] >= 3:
                fraud_score += 20
            if "Tingkat Kerusakan Barang: RINGAN" in damage_result:
                fraud_score += 40
            elif "Tingkat Kerusakan Barang: SEDANG" in damage_result:
                fraud_score += 15

            if fraud_score >= 70:
                fraud_level = "HIGH"
            elif fraud_score >= 40:
                fraud_level = "MEDIUM"
            else:
                fraud_level = "LOW"

            if fraud_level == "HIGH":
                complaint_status = "REJECTED"
                compensation_result = "❌ Mohon maaf sebesar-besarnya Kak, untuk saat ini komplain belum dapat kami setujui secara otomatis karena sistem kami memerlukan pengecekan manual secara lebih mendetail.\nNamun jangan khawatir, kami akan segera menghubungkan Kakak dengan tim Customer Service kami untuk bantuan lebih lanjut.\nSekali lagi, kami mohon maaf atas ketidaknyamanan ini 🙏😊"
            elif fraud_level == "MEDIUM":
                complaint_status = "NEED VERIFICATION"
                if "Tingkat Kerusakan Barang: RINGAN" in damage_result:
                    compensation_result = "Kami sangat memahami kekecewaan Kakak 🙏\nSebagai permohonan maaf kami, kami ingin memberikan voucher diskon belanja sebesar 10% untuk Kakak.\nVoucher ini akan segera kami proses dalam 2-3 hari kerja ya Kak. Terima kasih banyak atas kesabaran dan pengertiannya 🙏😊"
                else:
                    compensation_result = "📋 Untuk memastikan solusi terbaik, komplain ini masih memerlukan pengecekan manual oleh tim kami."
            else:
                complaint_status = "APPROVED"
                if "Tingkat Kerusakan Barang: RINGAN" in damage_result:
                    compensation_result = "Kami sangat memahami kekecewaan Kakak 🙏\nSebagai bentuk permohonan maaf, kami ingin memberikan voucher diskon belanja sebesar 10% untuk Kakak.\nVoucher ini akan segera kami proses dalam 2-3 hari kerja. Terima kasih banyak atas kesabaran Kakak 🙏😊"
                elif "Tingkat Kerusakan Barang: SEDANG" in damage_result:
                    compensation_result = "Kami sangat menyesal atas kejadian ini Kak 🙏\nSebagai bentuk tanggung jawab kami, kami akan mengembalikan 50% dari dana Kakak (partial refund).\nProses ini akan memakan waktu 2-3 hari kerja ya Kak. Terima kasih atas pengertiannya 🙏😊"
                elif "Tingkat Kerusakan Barang: BERAT" in damage_result:
                    compensation_result = "Kami sangat memohon maaf atas kerusakan parah pada barang Kakak 🙏 Karena video unboxing telah diverifikasi, silakan pilih opsi kompensasi yang paling nyaman untuk Kakak di bawah ini 👇"

            customer_response = f"""
Halo Kak {customer_name},

Berdasarkan hasil analisa dari foto dan video Kakak, kami menemukan:
**{damage_result}**

{compensation_result}

Terima kasih banyak sudah menyempatkan waktu untuk mengunggah video unboxing-nya ya Kak 🙏😊
"""
            st.write(customer_response)

            if complaint_status == "APPROVED" and "Tingkat Kerusakan Barang: BERAT" in damage_result:
                st.divider()
                st.write("**💳 Pilih Opsi Kompensasi Kakak**")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("💰 Full Refund (Pengembalian Dana Penuh)", key="refund_btn_with_video", use_container_width=True):
                        notify_store_compensation(customer_name, order_id, complaint_item, damage_result, fraud_level, "APPROVED", "Full Refund")
                        st.session_state.compensation_choice = "Full Refund"
                        st.session_state.compensation_notified = True
                        st.session_state.final_message = (
                            "✅ Pilihan kompensasi Kakak telah kami sampaikan ke pihak gudang. "
                            "Kami akan memproses Full Refund Kakak dalam 2-3 hari kerja. "
                            "Terima kasih banyak atas kesabaran dan kebaikannya Kak 🙏😊"
                        )
                        st.session_state.show_reset_button = True

                with col2:
                    if st.button("📦 Pengiriman Ulang Barang Baru", key="reshipping_btn_with_video", use_container_width=True):
                        notify_store_compensation(customer_name, order_id, complaint_item, damage_result, fraud_level, "APPROVED", "Pengiriman Ulang Barang Baru")
                        st.session_state.compensation_choice = "Pengiriman Ulang Barang Baru"
                        st.session_state.compensation_notified = True
                        st.session_state.final_message = (
                            "✅ Pilihan kompensasi Kakak telah kami sampaikan ke pihak gudang. "
                            "Kami akan memproses pengiriman ulang barang baru untuk Kakak dalam 2-3 hari kerja. "
                            "Terima kasih banyak atas kesabaran dan kebaikannya Kak 🙏😊"
                        )
                        st.session_state.show_reset_button = True
                        
                if st.session_state.compensation_notified:
                    st.divider()
                    st.success(f"✅ Opsi yang Kakak pilih: **{st.session_state.compensation_choice}**")
                    st.info(st.session_state.final_message)

            if st.session_state.show_reset_button:
                st.divider()
                st.write("\n")
                if st.button("🔄 Kembali ke Halaman Utama", key="reset_final_btn"):
                    reset_app_state()

    elif st.session_state.show_video_uploader and not st.session_state.video_checked:
        with st.chat_message("assistant"):
            st.warning(
                "Jika Kakak kebetulan tidak merekam video unboxing saat paket dibuka, jangan khawatir Kak, kami bisa membantu mengalihkan komplain ini ke tim Customer Service kami yang ramah 😊"
            )

            contact_cs_button = st.button(
                "📞 Hubungkan Saya dengan Customer Service",
                key="contact_cs_btn"
            )

            if contact_cs_button:
                if not customer_name or not order_id:
                    st.error("Mohon isi data lengkapnya terlebih dahulu ya Kak.")
                else:
                    notify_customer_service(
                        customer_name,
                        order_id,
                        complaint_item,
                        st.session_state.damage_result,
                        customer_data
                    )
                    st.session_state.contacted_customer_service = True
                    st.session_state.final_message = (
                        "✅ Kami telah menyampaikan kendala Kakak ke tim Customer Service kami. "
                        "Mohon ditunggu sebentar ya Kak, salah satu rekan kami akan segera menghubungi Kakak untuk membantu lebih lanjut. "
                        "Terima kasih banyak atas kesabarannya 🙏😊"
                    )
                    st.session_state.show_reset_button = True

            if st.session_state.contacted_customer_service:
                st.divider()
                st.success("✅ Permintaan bantuan sudah terkirim ke tim Customer Service.")
                st.info(st.session_state.final_message)
                if st.session_state.show_reset_button:
                    st.divider()
                    st.write("\n")
                    if st.button("🔄 Kembali ke Halaman Utama", key="reset_final_btn_cs"):
                        reset_app_state()
