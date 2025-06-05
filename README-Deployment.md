# 🚀 Panduan Deploy ke Vercel

## 📋 Prasyarat
1. Akun GitHub
2. Akun Vercel (gratis)
3. File project sudah di-commit ke GitHub

## 🔧 Cara Deploy

### Metode 1: Deploy via GitHub (Recommended)

1. **Push ke GitHub Repository**
   ```bash
   git add .
   git commit -m "Ready for Vercel deployment"
   git push origin main
   ```

2. **Connect ke Vercel**
   - Kunjungi [vercel.com](https://vercel.com)
   - Login dengan akun GitHub
   - Klik "New Project"
   - Import repository GitHub Anda
   - Vercel akan auto-detect sebagai Python project

3. **Configure Build Settings**
   - Framework Preset: `Other`
   - Build Command: `pip install -r requirements-vercel.txt`
   - Output Directory: `.`
   - Install Command: (biarkan kosong)

### Metode 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login ke Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel --prod
   ```

## 📁 Struktur File untuk Deployment

```
project/
├── app.py                 # Main Flask app
├── vercel.json           # Vercel configuration
├── requirements-vercel.txt # Minimal dependencies
├── templates/
│   └── index.html        # HTML template
├── static/
│   ├── img/             # Images
│   └── ...
└── README-Deployment.md  # This file
```

## ⚙️ File Konfigurasi

### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "./app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "./app.py"
    }
  ]
}
```

### requirements-vercel.txt
File ini berisi dependency minimal untuk deployment yang lebih cepat.

## 🔍 Troubleshooting

### Error: Build terlalu lama
- Gunakan `requirements-vercel.txt` instead of `requirements.txt`
- Hapus dependencies yang tidak perlu untuk deployment

### Error: Static files tidak load
- Pastikan folder `static/` ada di root project
- Check path gambar di HTML menggunakan `static/img/...`

### Error: Template not found
- Pastikan folder `templates/` ada di root project
- Check `render_template('index.html')` di app.py

## 🌟 Tips Optimasi

1. **Gunakan CDN untuk libraries**
   - Chart.js, AOS, Tailwind sudah load dari CDN
   - Tidak perlu install local

2. **Compress Images**
   - Optimasi gambar di folder `static/img/`
   - Gunakan format WebP jika memungkinkan

3. **Environment Variables**
   - Untuk production, set `FLASK_ENV=production`

## 📞 Support

Jika ada error saat deployment, cek:
- Vercel deployment logs
- Pastikan semua file sudah di-commit
- Verifikasi struktur folder sesuai panduan

Happy Deploying! 🎉 