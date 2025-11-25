document.addEventListener('DOMContentLoaded', () => {
    const customSelect = document.getElementById('customSelect');
    const customSelectOptions = document.getElementById('customSelectOptions');
    const selectedFlag = document.getElementById('selectedFlag');
    const selectedLanguage = document.getElementById('selectedLanguage');
    let selectedValue = 'Egypt';

    customSelect.addEventListener('click', () => {
        customSelectOptions.style.display =
            customSelectOptions.style.display === 'block' ? 'none' : 'block';
    });

    customSelectOptions.addEventListener('click', event => {
        const option = event.target.closest('div[data-value]');
        if (option) {
            selectedValue = option.getAttribute('data-value');
            selectedFlag.src = option.getAttribute('data-flag');
            selectedLanguage.textContent = selectedValue;
            customSelectOptions.style.display = 'none';
        }
    });

    window.addEventListener('click', event => {
        if (!customSelect.contains(event.target)) {
            customSelectOptions.style.display = 'none';
        }
    });

    const uploadBtn = document.getElementById('uploadBtn');
    const startRecordBtn = document.getElementById('startRecordBtn');
    const stopRecordBtn = document.getElementById('stopRecordBtn');
    const repeatRecordBtn = document.getElementById('repeatRecordBtn');
    const convertBtn = document.getElementById('convertBtn');
    const videoElement = document.getElementById('video');
    const resultText = document.getElementById('resultText');

    let videoFile = null;
    let recorder = null;
    let stream = null;

    function enableConvertButton() {
        convertBtn.classList.remove('bg-gray-500', 'cursor-not-allowed');
        convertBtn.classList.add('bg-blue-500', 'hover:bg-blue-700');
        convertBtn.disabled = false;
    }

    
    function stopRecordState() {
        startRecordBtn.disabled = false;
        stopRecordBtn.disabled = true;
        repeatRecordBtn.disabled = true;
        videoElement.srcObject = null;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    function resetRecordingState() {
        startRecordBtn.disabled = false;
        stopRecordBtn.disabled = true;
        repeatRecordBtn.disabled = true;
        videoElement.srcObject = null;
        
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    
        if (recorder) {
            recorder.destroy();
            recorder = null;
        }
    
        videoFile = null;
        recordedBlob = null;
    
        // Clear video element source
        videoElement.src = "";
    }
    
    uploadBtn.addEventListener('click', () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'video/*';
        input.onchange = e => {
            const file = e.target.files[0];
            if (file) {
                videoFile = file;
                const url = URL.createObjectURL(file);
                videoElement.src = url;
                videoElement.load();
                enableConvertButton();
            }
        };
        input.click();
    });

    startRecordBtn.addEventListener('click', async () => {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        videoElement.play();

        recorder = RecordRTC(stream, {
            type: 'video',
            mimeType: 'video/mp4',
            frameRate: 30,
            width: 640,
            height: 480,
            bitrate: 128000
        });

        recorder.startRecording();
        startRecordBtn.disabled = true;
        stopRecordBtn.disabled = false;
        repeatRecordBtn.disabled = true;
    });

    stopRecordBtn.addEventListener('click', () => {
        recorder.stopRecording(() => {
            const blob = recorder.getBlob();
            videoFile = new File([blob], 'recorded-video.mp4', { type: 'video/mp4' });
            const url = URL.createObjectURL(blob);
            videoElement.srcObject = null;
            videoElement.src = url;
            videoElement.load();
            enableConvertButton();
        });
        stopRecordState();
    });

    repeatRecordBtn.addEventListener('click', () => {
        resetRecordingState();
    });

    convertBtn.addEventListener('click', async () => {
        if (!videoFile) {
            alert('Please upload or record a video first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', videoFile);
        formData.append('model', selectedValue);

        convertBtn.textContent = 'Loading...';
        convertBtn.disabled = true;

        try {
            const response = await fetch('http://127.0.0.1:5001/flask-api', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            resultText.innerText = result.result;

            convertBtn.textContent = 'Translate';
            convertBtn.disabled = false;
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to convert video.');
            convertBtn.textContent = 'Translate';
            convertBtn.disabled = false;
        }
    });
});
