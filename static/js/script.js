document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('cameraFeed');
    const captureBtn = document.getElementById('captureBtn');
    const canvas = document.getElementById('canvas');
    const countdownDiv = document.getElementById('countdown');
    let stream;

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
    }

    function capturePhoto() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        stopCamera();
        const imageDataURL = canvas.toDataURL('image/png');
        const filename = `photo_${Date.now()}.png`;

        // Kirim gambar ke server tanpa menyimpan langsung di sini
        const formData = new FormData();
        formData.append('filename', filename);

        fetch('/capture', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.redirected) {
                window.location.href = response.url;
            }
        })
        .catch(error => {
            console.error('Error sending captured image:', error);
        });

        // Secara konseptual, di sini Anda akan menyimpan imageDataURL ke server
        // atau langsung memprosesnya. Untuk contoh ini, kita akan me-redirect
        // ke halaman editor dengan nama file sementara.
        // Anda perlu implementasi penyimpanan gambar di sisi server yang sebenarnya.
    }

    let countdownInterval;
    let countdownTime = 5;

    function startCountdown() {
        countdownTime = 5;
        countdownDiv.innerText = countdownTime;
        countdownInterval = setInterval(() => {
            countdownTime--;
            countdownDiv.innerText = countdownTime;
            if (countdownTime <= 0) {
                clearInterval(countdownInterval);
                countdownDiv.innerText = '';
                capturePhoto();
            }
        }, 1000);
    }

    // Implementasi deteksi tangan (placeholder - memerlukan library ML/CV)
    function detectHand() {
        // Secara periodik ambil frame dari video dan lakukan deteksi tangan
        // Gunakan library seperti TensorFlow.js dengan model handpose atau
        // integrasikan dengan backend yang melakukan deteksi (lebih kompleks)

        // Contoh sederhana (selalu false):
        return false;
    }

    function processFrame() {
        if (video.srcObject) {
            // Jika deteksi tangan "hi five" berhasil
            if (detectHand()) {
                startCountdown();
                // Nonaktifkan deteksi sementara setelah deteksi
                clearInterval(frameProcessingInterval);
            }
        }
    }

    let frameProcessingInterval;

    captureBtn.addEventListener('click', () => {
        startCountdown(); // Untuk demo tanpa deteksi tangan
    });

    startCamera();

    // Untuk mengaktifkan deteksi tangan (memerlukan implementasi detectHand)
    // frameProcessingInterval = setInterval(processFrame, 100);
});