<h1 align="center">Emotion Recognition System</h1>


<div style="background-color: #eaf2f8; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #3498db;">
  <h2 style="color: #2874a6; margin-top: 0;">🔧 Basic Setup</h2>
  
  <h3 style="color: #21618c;">1️⃣ Clone the repository</h3>
  <pre style="background-color: #f4f6f6; padding: 12px; border-radius: 6px; border: 1px solid #d6eaf8;">
<code>git clone https://github.com/bilinyang/Emotion_Dectector.git
cd Emotion_Detector</code></pre>
  
  <h3 style="color: #21618c; margin-top: 25px;">2️⃣ Create Conda Environment</h3>
  <pre style="background-color: #f4f6f6; padding: 12px; border-radius: 6px; border: 1px solid #d6eaf8;">
<code>conda create -n &lt;venv_name&gt; python=3.9.22
conda activate &lt;venv_name&gt;</code></pre>
  
  <h3 style="color: #21618c; margin-top: 25px;">3️⃣ Install Dependencies</h3>
  <div style="background-color: #f4f6f6; padding: 12px; border-radius: 6px; border: 1px solid #d6eaf8; margin-bottom: 10px;">
    <strong>Core Requirements:</strong>
    <ul style="margin-top: 5px;">
      <li>OpenCV 4.5+</li>
      <li>TensorFlow 2.6+</li>
      <li>Other packages from requirements.txt</li>
    </ul>
  </div>
  
  <pre style="background-color: #f4f6f6; padding: 12px; border-radius: 6px; border: 1px solid #d6eaf8;">
<code>pip install -r requirements.txt</code></pre>
  
  <div style="background-color: #d5f5e3; padding: 12px; border-radius: 6px; margin-top: 15px; border-left: 4px solid #28b463;">
    <strong>💡 Conda Tip:</strong> For GPU support, install TensorFlow with Conda:
    <pre style="background-color: #e8f8f5; padding: 10px; border-radius: 5px; margin-top: 8px;">
<code>conda install -c conda-forge tensorflow-gpu</code></pre>
  </div>
</div>





<h1 style="visibility: hidden;">&nbsp;</h1>
<h2 align="center" style="color: #2e86c1;">🚀 Quick Start Guide</h2>

<div style="background-color: #f8f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #2e86c1;">
  <strong>Prerequisites:</strong>
  <ul>
    <li>You are in the project root directory</li>
    <li>Environment is set up (<code>requirements.txt</code> installed)</li>
  </ul>
</div>

<hr style="border: 1px solid #ddd;">

<h1 style="visibility: hidden;">&nbsp;</h1>
<h3 style="color: #2874a6;">1️⃣ Train a Model</h3>

<h4>Option A: Specify the Model You Want to Train</h4>
<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/train_emotion_detector.py -m &lt;model_name&gt;</code></pre>

<h4>Available Models:</h4>
<ul style="list-style-type: square;">
  <li><code>emotionvggnet</code> (default)</li>
  <li><code>lenet</code></li>
  <li><code>minigooglenet</code></li>
  <li><code>minivggnet</code></li>
  <li><code>shallownet</code></li>
</ul>

<h4>Option B: Train Default Model (emotionvggnet)</h4>
<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/train_emotion_detector.py</code></pre>

<div style="background-color: #e8f8f5; padding: 10px; border-radius: 5px; margin-top: 10px;">
  <strong>📁 Outputs:</strong>
  <ul>
    <li>Trained models: <code>emo_rec/built_models/</code></li>
    <li>Training logs: <code>emo_rec/training_logs/</code></li>
  </ul>
</div>

<hr style="border: 1px solid #ddd;">





<h1 style="visibility: hidden;">&nbsp;</h1>
<h3 style="color: #2874a6;">2️⃣ Test a Model</h3>

<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/test_emotion_detector.py -m &lt;model_name&gt;</code></pre>

<div style="background-color: #fdedec; padding: 10px; border-radius: 5px; border-left: 4px solid #e74c3c;">
  <strong>❗ Requirements:</strong>
  <ul>
    <li>Model must be trained first (Step 1)</li>
    <li><code>-m</code> flag is mandatory</li>
  </ul>
</div>

<h4>Example:</h4>
<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/test_emotion_detector.py -m minivggnet</code></pre>

<hr style="border: 1px solid #ddd;">





<h1></h1>
<h3 style="color: #2874a6;">3️⃣ Run Emotion Detection</h3>

<h4>Option A: Process Video File</h4>
<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/run_emotion_detector.py -m &lt;model_name&gt; -v &lt;video_path&gt;</code></pre>

<h4>Example (to run the demo video already provided):</h4>
<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/run_emotion_detector.py -m emotionvggnet -v emo_rec/video/example.mp4</code></pre>

<h4>Option B: Real-Time Webcam</h4>
<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/run_emotion_detector.py -m &lt;model_name&gt;</code></pre>

<div style="background-color: #ebf5fb; padding: 10px; border-radius: 5px; margin-top: 10px;">
  <strong>🖥️ Controls:</strong>
  <ol>
    <li>Select the camera window</li>
    <li>Press:
      <ul>
        <li>Mac: <kbd>Cmd</kbd> + <kbd>Q</kbd></li>
        <li>Windows: <kbd>Ctrl</kbd> + <kbd>Q</kbd></li>
      </ul>
    </li>
  </ol>
</div>

<h4>Example:</h4>
<pre style="background-color: #f4f6f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
<code>python emo_rec/run_emotion_detector.py -m lenet</code></pre>

<div style="background-color: #fdedec; padding: 10px; border-radius: 5px; border-left: 4px solid #e74c3c; margin-top: 15px;">
  <strong>🔴 Important:</strong>
  <ul>
    <li>Model must be trained first (Step 1)</li>
    <li><code>-m &lt;model_name&gt;</code> is always required</li>
    <li>Uses default camera device</li>
  </ul>
</div>
