# PanganCast Backend — Panduan Deploy

## Struktur File
```
PanganCast_flask/
├── app.py
├── core.py
├── requirements.txt
├── Procfile              ← untuk Railway / Heroku
├── gunicorn.conf.py      ← konfigurasi Gunicorn
└── models/               ← folder model tersimpan (auto-created)
```

---

## Opsi 1: Railway (Rekomendasi, Gratis)

1. Buat akun di https://railway.app
2. Klik "New Project" → "Deploy from GitHub repo"
3. Push folder `PanganCast_flask` ke GitHub
4. Railway otomatis detect `Procfile` dan deploy

File `Procfile` sudah disertakan:
```
web: gunicorn app:app -c gunicorn.conf.py
```

Setelah deploy, Railway beri URL seperti:
```
https://pangancast-backend.up.railway.app
```

Ganti `kBaseUrl` di Flutter:
```dart
// lib/services/api_service.dart
const String kBaseUrl = 'https://pangancast-backend.up.railway.app';
```

---

## Opsi 2: Render (Gratis)

1. Buat akun di https://render.com
2. New → Web Service → Connect GitHub repo
3. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app -c gunicorn.conf.py`
4. Deploy

---

## Opsi 3: VPS / Server Sendiri

```bash
# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Jalankan dengan Gunicorn (production)
gunicorn app:app -c gunicorn.conf.py

# Atau dengan systemd service (agar auto-start)
# Buat file: /etc/systemd/system/pangancast.service
```

Contoh systemd service:
```ini
[Unit]
Description=PanganCast Flask Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/PanganCast_flask
ExecStart=/usr/bin/gunicorn app:app -c gunicorn.conf.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable pangancast
sudo systemctl start pangancast
```

---

## Opsi 4: Lokal (Development)

```bash
cd PanganCast_flask
pip install -r requirements.txt
python app.py
# Server berjalan di http://localhost:5000
```

---

## Catatan Penting

- Folder `models/` harus bisa ditulis oleh server (writable)
- Untuk production, tambahkan Nginx sebagai reverse proxy
- Session disimpan in-memory — restart server = session hilang
- Untuk persistent session, pertimbangkan Redis

---

## Test Endpoint Setelah Deploy

```bash
# Health check
curl https://YOUR_URL/health

# Upload dataset
curl -X POST https://YOUR_URL/upload \
  -F "file=@dataset.csv" \
  -F "date_column=Tanggal" \
  -F "commodity_column=Harga"
```
