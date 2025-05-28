document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const predictionSpan = document.getElementById('prediction');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Set up drawing context
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Drawing event listeners
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        
        const [currentX, currentY] = getCoordinates(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();

        [lastX, lastY] = [currentX, currentY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const x = e.type.includes('touch') 
            ? e.touches[0].clientX - rect.left 
            : e.clientX - rect.left;
        const y = e.type.includes('touch') 
            ? e.touches[0].clientY - rect.top 
            : e.clientY - rect.top;
        return [x, y];
    }

    // Add event listeners for both mouse and touch events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startDrawing(e);
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        draw(e);
    });
    canvas.addEventListener('touchend', stopDrawing);

    // Clear canvas
    clearBtn.addEventListener('click', function() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        predictionSpan.textContent = '-';
    });

    // Predict
    predictBtn.addEventListener('click', async function() {
        const imageData = canvas.toDataURL('image/png');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();
            
            if (data.error) {
                predictionSpan.textContent = 'Error';
                console.error(data.error);
            } else {
                predictionSpan.textContent = data.prediction;
            }
        } catch (error) {
            predictionSpan.textContent = 'Error';
            console.error('Error:', error);
        }
    });
}); 