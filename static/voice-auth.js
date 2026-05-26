class VoiceAuthenticator {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.enrollmentRecordings = [];
        this.currentRecording = null;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.animationId = null;
        this.audioStream = null; // Store the stream for reuse
        this.currentChallengeNonce = null;

        this.init();
        
        // Clean up stream when page unloads
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    cleanup() {
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }
    }

    init() {
        this.setupEventListeners();
        this.checkMicrophonePermission();
        this.createWaveVisualization();
    }

    setupEventListeners() {
        // Enrollment
        document.getElementById('startEnrollBtn').addEventListener('click', () => this.startRecording('enroll'));
        document.getElementById('stopEnrollBtn').addEventListener('click', () => this.stopRecording('enroll'));
        document.getElementById('submitEnrollBtn').addEventListener('click', () => this.submitEnrollment());

        // Verification
        document.getElementById('startVerifyBtn').addEventListener('click', () => this.startRecording('verify'));
        document.getElementById('stopVerifyBtn').addEventListener('click', () => this.stopRecording('verify'));
        document.getElementById('submitVerifyBtn').addEventListener('click', () => this.submitVerification());

        // Passphrase
        document.getElementById('startPassphraseBtn').addEventListener('click', () => this.startRecording('passphrase'));
        document.getElementById('stopPassphraseBtn').addEventListener('click', () => this.stopRecording('passphrase'));
        document.getElementById('submitPassphraseBtn').addEventListener('click', () => this.submitPassphrase());
    }

    async checkMicrophonePermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            this.showStatus('Microphone access granted ✅', 'success');
        } catch (error) {
            this.showStatus('❌ Microphone access denied. Please allow microphone permissions.', 'error');
        }
    }

    createWaveVisualization() {
        // Create wave bars for visualization
        const enrollBars = document.getElementById('enrollWaveBars');
        const verifyBars = document.getElementById('verifyWaveBars');
        
        for (let i = 0; i < 32; i++) {
            const bar1 = document.createElement('div');
            bar1.className = 'wave-bar';
            enrollBars.appendChild(bar1);

            const bar2 = document.createElement('div');
            bar2.className = 'wave-bar';
            verifyBars.appendChild(bar2);
        }
    }

    async startRecording(type) {
        try {
            // Reuse existing stream if available and active
            let stream = this.audioStream;
            if (!stream || !stream.active) {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                this.audioStream = stream; // Store for reuse
            }

            this.setupAudioAnalysis(stream);
            
            // Try different MIME types for better compatibility
            let mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/wav';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = ''; // Use default
                    }
                }
            }
            
            console.log('Using MediaRecorder mimeType:', mimeType);
            this.mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});

            this.audioChunks = [];
            this.isRecording = true;

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording(type);
                // DON'T stop the stream tracks - keep them for reuse
                // stream.getTracks().forEach(track => track.stop());
                this.stopVisualization();
            };

            this.mediaRecorder.start();
            this.updateUI(type, 'recording');
            this.startVisualization(type);
            this.showStatus(`🎤 Recording ${type}... Speak clearly!`, 'info');

        } catch (error) {
            console.error('Error starting recording:', error);
            
            // Clean up any failed stream
            if (this.audioStream && !this.audioStream.active) {
                this.audioStream = null;
            }
            
            // More specific error message
            if (error.name === 'NotAllowedError') {
                this.showStatus('❌ Microphone permission denied. Please allow microphone access and refresh the page.', 'error');
            } else if (error.name === 'NotFoundError') {
                this.showStatus('❌ No microphone found. Please connect a microphone.', 'error');
            } else {
                this.showStatus('❌ Failed to start recording. Check microphone permissions and try refreshing the page.', 'error');
            }
        }
    }

    stopRecording(type) {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateUI(type, 'stopped');
        }
    }

    setupAudioAnalysis(stream) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        const source = this.audioContext.createMediaStreamSource(stream);
        
        source.connect(this.analyser);
        this.analyser.fftSize = 64;
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    }

    startVisualization(type) {
        const waveContainer = document.getElementById(`${type}Wave`);
        const waveBars = document.getElementById(`${type}WaveBars`);
        
        waveContainer.style.display = 'flex';

        const animate = () => {
            if (!this.isRecording) return;

            this.analyser.getByteFrequencyData(this.dataArray);
            const bars = waveBars.children;

            for (let i = 0; i < bars.length; i++) {
                const value = this.dataArray[i] || 0;
                const height = (value / 255) * 40 + 4;
                bars[i].style.height = `${height}px`;
            }

            this.animationId = requestAnimationFrame(animate);
        };

        animate();
    }

    stopVisualization() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }

    async processRecording(type) {
        console.log('Processing recording, audioChunks:', this.audioChunks.length);
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        console.log('Created audioBlob:', audioBlob.size, 'bytes, type:', audioBlob.type);
        
        // Convert to WAV format
        const wavBlob = await this.convertToWav(audioBlob);
        console.log('Converted to WAV:', wavBlob.size, 'bytes, type:', wavBlob.type);
        
        if (type === 'enroll') {
            this.enrollmentRecordings.push(wavBlob);
            this.updateEnrollmentList();
            this.showStatus(`✅ Enrollment recording ${this.enrollmentRecordings.length} saved!`, 'success');
            console.log('Added to enrollmentRecordings, total:', this.enrollmentRecordings.length);
        } else {
            this.currentRecording = wavBlob;
            this.showStatus(`✅ ${type} recording saved!`, 'success');
        }
    }

    async convertToWav(audioBlob) {
        // Create audio element to decode
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        return new Promise((resolve) => {
            audio.addEventListener('loadeddata', async () => {
                try {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const response = await fetch(audioUrl);
                    const arrayBuffer = await response.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    // Convert to 16kHz mono WAV
                    const wavBuffer = this.audioBufferToWav(audioBuffer);
                    const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                    
                    URL.revokeObjectURL(audioUrl);
                    resolve(wavBlob);
                } catch (error) {
                    console.error('Conversion error:', error);
                    // Fallback: use original blob
                    resolve(audioBlob);
                }
            });
        });
    }

    audioBufferToWav(audioBuffer) {
        // Resample to 16kHz mono
        const targetSampleRate = 16000;
        const sourceSampleRate = audioBuffer.sampleRate;
        const channels = 1; // Mono
        
        // Get left channel or mix down to mono
        let audioData;
        if (audioBuffer.numberOfChannels === 1) {
            audioData = audioBuffer.getChannelData(0);
        } else {
            // Mix down to mono
            const left = audioBuffer.getChannelData(0);
            const right = audioBuffer.getChannelData(1);
            audioData = new Float32Array(left.length);
            for (let i = 0; i < left.length; i++) {
                audioData[i] = (left[i] + right[i]) / 2;
            }
        }

        // Resample if needed
        if (sourceSampleRate !== targetSampleRate) {
            const resampleRatio = targetSampleRate / sourceSampleRate;
            const newLength = Math.round(audioData.length * resampleRatio);
            const resampled = new Float32Array(newLength);
            
            for (let i = 0; i < newLength; i++) {
                const sourceIndex = i / resampleRatio;
                const index = Math.floor(sourceIndex);
                const fraction = sourceIndex - index;
                
                if (index + 1 < audioData.length) {
                    resampled[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
                } else {
                    resampled[i] = audioData[index];
                }
            }
            audioData = resampled;
        }

        // Convert to 16-bit PCM
        const pcmData = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            const sample = Math.max(-1, Math.min(1, audioData[i]));
            pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }

        // Create WAV file
        const buffer = new ArrayBuffer(44 + pcmData.length * 2);
        const view = new DataView(buffer);

        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + pcmData.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, channels, true);
        view.setUint32(24, targetSampleRate, true);
        view.setUint32(28, targetSampleRate * channels * 2, true);
        view.setUint16(32, channels * 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, pcmData.length * 2, true);

        // PCM data
        const pcmView = new Int16Array(buffer, 44);
        pcmView.set(pcmData);

        return buffer;
    }

    updateUI(type, state) {
        const startBtn = document.getElementById(`start${type.charAt(0).toUpperCase() + type.slice(1)}Btn`);
        const stopBtn = document.getElementById(`stop${type.charAt(0).toUpperCase() + type.slice(1)}Btn`);
        const submitBtn = document.getElementById(`submit${type.charAt(0).toUpperCase() + type.slice(1)}Btn`);
        const indicator = document.getElementById(`${type}Indicator`);

        if (state === 'recording') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            submitBtn.disabled = true;
            indicator.classList.add('active');
        } else if (state === 'stopped') {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            submitBtn.disabled = false;
            indicator.classList.remove('active');
        }
    }

    updateEnrollmentList() {
        const list = document.getElementById('enrollRecordings');
        list.innerHTML = '';

        this.enrollmentRecordings.forEach((recording, index) => {
            const item = document.createElement('div');
            item.className = 'recording-item';
            item.innerHTML = `
                <span>Recording ${index + 1}</span>
                <button class="btn btn-danger" onclick="voiceAuth.removeEnrollment(${index})" style="padding: 5px 10px; margin: 0;">❌</button>
            `;
            list.appendChild(item);
        });

        const submitBtn = document.getElementById('submitEnrollBtn');
        submitBtn.disabled = this.enrollmentRecordings.length < 2;
    }

    removeEnrollment(index) {
        this.enrollmentRecordings.splice(index, 1);
        this.updateEnrollmentList();
    }

    async submitEnrollment() {
        const userId = document.getElementById('userId').value.trim();
        if (!userId) {
            this.showStatus('❌ Please enter a User ID', 'error');
            return;
        }

        if (this.enrollmentRecordings.length < 2) {
            this.showStatus('❌ Please record at least 2 voice samples', 'error');
            return;
        }

        console.log('Starting enrollment with recordings:', this.enrollmentRecordings.length);
        this.enrollmentRecordings.forEach((recording, index) => {
            console.log(`Recording ${index}: size=${recording.size}, type=${recording.type}`);
        });

        this.showProgress(0);
        const formData = new FormData();

        this.enrollmentRecordings.forEach((recording, index) => {
            console.log(`Adding recording ${index} to FormData: ${recording.size} bytes`);
            formData.append('files', recording, `enrollment_${index}.wav`);
        });

        // Debug FormData contents
        for (let pair of formData.entries()) {
            console.log('FormData entry:', pair[0], pair[1]);
        }

        try {
            // Send enrollment request directly
            const response = await fetch(`/enroll/${userId}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.hideProgress();

            if (response.ok) {
                this.showStatus(`✅ Enrollment successful! Processed ${result.samples_processed} samples.`, 'success');
                this.enrollmentRecordings = [];
                this.updateEnrollmentList();
            } else {
                this.showStatus(`❌ Enrollment failed: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.hideProgress();
            this.showStatus('❌ Network error during enrollment', 'error');
        }
    }

    async submitVerification() {
        const userId = document.getElementById('userId').value.trim();
        if (!userId) {
            this.showStatus('❌ Please enter a User ID', 'error');
            return;
        }

        if (!this.currentRecording) {
            this.showStatus('❌ Please record a voice sample first', 'error');
            return;
        }

        this.showProgress(0);
        const formData = new FormData();
        formData.append('file', this.currentRecording, 'verification.wav');

        try {
            const response = await fetch(`/verify/${userId}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.hideProgress();

            if (response.ok) {
                if (result.verified) {
                    this.showStatus(`✅ Verification successful! Score: ${result.score}`, 'success');
                } else if (result.challenge_required) {
                    // Server returns challenge_text, not passphrase_words
                    this.showPassphraseChallenge(result.challenge_text, result.challenge_nonce);
                } else {
                    this.showStatus(`❌ Verification failed. Score: ${result.score}`, 'error');
                }
            } else {
                this.showStatus(`❌ Verification error: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.hideProgress();
            this.showStatus('❌ Network error during verification', 'error');
        }
    }

    showPassphraseChallenge(challengeText, challengeNonce) {
        const challengeDiv = document.getElementById('passphraseChallenge');
        const wordsDiv = document.getElementById('challengeWords');
        
        // Store the nonce for later use
        this.currentChallengeNonce = challengeNonce;
        
        // Display the challenge text (e.g., "536203 advocate excellent")
        wordsDiv.textContent = challengeText;
        challengeDiv.style.display = 'block';
        
        this.showStatus('🔐 Passphrase challenge required. Please record the text shown above.', 'info');
    }

    async submitPassphrase() {
        const userId = document.getElementById('userId').value.trim();
        if (!this.currentRecording) {
            this.showStatus('❌ Please record the passphrase first', 'error');
            return;
        }

        this.showProgress(0);
        const formData = new FormData();
        formData.append('file', this.currentRecording, 'passphrase.wav');

        try {
            const headers = {};
            if (this.currentChallengeNonce) {
                headers['X-Nonce'] = this.currentChallengeNonce;
            }
            
            const response = await fetch(`/verify_passphrase/${userId}`, {
                method: 'POST',
                headers: headers,
                body: formData
            });

            const result = await response.json();
            this.hideProgress();

            if (response.ok) {
                if (result.verified) {
                    this.showStatus(`✅ Two-factor verification successful! Score: ${result.score}`, 'success');
                    document.getElementById('passphraseChallenge').style.display = 'none';
                } else {
                    this.showStatus(`❌ Passphrase verification failed: ${result.message}`, 'error');
                }
            } else {
                this.showStatus(`❌ Passphrase error: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.hideProgress();
            this.showStatus('❌ Network error during passphrase verification', 'error');
        }
    }

    showStatus(message, type) {
        const statusDiv = document.getElementById('statusDisplay');
        statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.innerHTML = '';
            }, 5000);
        }
    }

    showProgress(percent) {
        const progressDiv = document.getElementById('uploadProgress');
        const progressBar = document.getElementById('progressBar');
        
        progressDiv.style.display = 'block';
        progressBar.style.width = `${percent}%`;
    }

    hideProgress() {
        const progressDiv = document.getElementById('uploadProgress');
        progressDiv.style.display = 'none';
    }
}

// Initialize the voice authenticator when the page loads
const voiceAuth = new VoiceAuthenticator();

// Make it globally available for button onclick handlers
window.voiceAuth = voiceAuth;
