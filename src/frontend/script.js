async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Select image first");
        return;
    }

    document.getElementById("loading").innerText = "Predicting...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict-image", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        document.getElementById("loading").innerText = "";

        if (!response.ok) {
            const msg = data.detail || data.error || "Prediction failed";
            document.getElementById("result").innerText = `Error: ${msg}`;
            document.getElementById("top3").innerHTML = "";
            return;
        }

        const resultData = data.prediction && typeof data.prediction === "object"
            ? data.prediction
            : data;

        if (
            typeof resultData.prediction === "undefined" ||
            typeof resultData.confidence !== "number" ||
            !Array.isArray(resultData.top3)
        ) {
            document.getElementById("result").innerText = "Unexpected response from server";
            document.getElementById("top3").innerHTML = "";
            return;
        }

        document.getElementById("result").innerText =
            `Prediction: ${resultData.prediction} (${(resultData.confidence * 100).toFixed(1)}%)`;

        const top3Div = document.getElementById("top3");
        top3Div.innerHTML = "<h4>Top 3 Predictions</h4>";

        resultData.top3.forEach(item => {
            const percentValue = Number(item.confidence) * 100;
            const percent = percentValue.toFixed(1);

            top3Div.innerHTML += `
                <div class="bar-container">
                    <div class="bar-label">${item.digit} → ${percent}%</div>
                    <div class="bar">
                        <div class="bar-fill" style="width:${percentValue}%"></div>
                    </div>
                </div>
            `;
        });
    } catch (error) {
        document.getElementById("loading").innerText = "";
        document.getElementById("result").innerText = "Network or server error during prediction";
        document.getElementById("top3").innerHTML = "";
        console.error(error);
    }
}

function previewImage() {
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const file = fileInput.files[0];

    if (!file) {
        preview.src = "";
        preview.style.display = "none";
        return;
    }

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
}