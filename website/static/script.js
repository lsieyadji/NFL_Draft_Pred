document.getElementById('predict-btn').addEventListener('click', async function() {
    // Get all input elements
    const inputs = {
        position: document.getElementById('position'),
        height: document.getElementById('height'),
        weight: document.getElementById('weight'),
        bmi: document.getElementById('bmi'),
        age: document.getElementById('age'),
        sprint_40yd: document.getElementById('sprint_40yd'),
        vertical_jump: document.getElementById('vertical_jump'),
        bench_press: document.getElementById('bench_press'),
        broad_jump: document.getElementById('broad_jump')
    };

    // Validate all fields
    let allFilled = true;
    for (const [field, element] of Object.entries(inputs)) {
        if (!element.value.trim()) {
            allFilled = false;
            alert(`Please fill in the ${field.replace('_', ' ')} field`);
            return;
        }
    }

    // Prepare form data
    const formData = {
        position: inputs.position.value,
        height: parseFloat(inputs.height.value),
        weight: parseFloat(inputs.weight.value),
        bmi: parseFloat(inputs.bmi.value),
        age: parseInt(inputs.age.value),
        sprint_40yd: parseFloat(inputs.sprint_40yd.value),
        vertical_jump: parseFloat(inputs.vertical_jump.value),
        bench_press: parseInt(inputs.bench_press.value),
        broad_jump: parseFloat(inputs.broad_jump.value)
    };

    // Show loading state
    const button = this;
    button.textContent = 'Predicting...';
    button.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        if (result.error) throw new Error(result.error);
        displayResults(result);

    } catch (error) {
        alert('Prediction failed: ' + error.message);
    } finally {
        button.textContent = 'Predict Draft Selection';
        button.disabled = false;
    }
});

function displayResults(result) {
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('prediction-text');
    const probabilityDiv = document.getElementById('probability');
    const confidenceDiv = document.getElementById('confidence');

    if (result.drafted) {
        predictionText.textContent = '🎉 PREDICTED: DRAFTED';
        predictionText.className = 'drafted';
    } else {
        predictionText.textContent = '❌ PREDICTED: NOT DRAFTED';
        predictionText.className = 'not-drafted';
    }

    probabilityDiv.textContent = `Probability: ${result.probability}%`;
    confidenceDiv.textContent = `Confidence: ${result.confidence.toUpperCase()}`;
    resultDiv.classList.remove('hidden');
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}