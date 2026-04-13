// =========================
// DOM ELEMENTS
// =========================
const form = document.getElementById("uploadForm");
const graphDiv = document.getElementById("graph");
const explanationDiv = document.getElementById("explanation");
const loader = document.getElementById("loader");
const canvas = document.getElementById("videoCanvas");
const videoInput = document.querySelector('input[name="video"]');
const tabs = document.querySelectorAll(".tab");
const reportBtn = document.getElementById("generateReport");

// =========================
// STATE
// =========================
let currentData = null;
let selectedPoint = null;

// =========================
// UTILS
// =========================
function sanitizeHTML(str = "") {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, "<br>");
}

function getSeries(data) {
    if (!Array.isArray(data)) return { pos: null, vel: null, acc: null };

    const find = (key) =>
        data.find(t => (t?.name || "").toLowerCase().includes(key)) || null;

    return {
        pos: find("position"),
        vel: find("velocity"),
        acc: find("acceleration")
    };
}

// =========================
// GRAPH RENDERING
// =========================
function renderGraph(type = "all") {
    if (!currentData?.data) return;

    const { pos, vel, acc } = getSeries(currentData.data);

    let traces = [];
    let yTitle = "Value";

    if (type === "all") {
        traces = [pos, vel, acc].filter(Boolean);
    } else if (type === "pos" && pos) {
        traces = [{ ...pos, name: "Position (m)" }];
        yTitle = "Position (m)";
    } else if (type === "vel" && vel) {
        traces = [{ ...vel, name: "Velocity (m/s)" }];
        yTitle = "Velocity (m/s)";
    } else if (type === "acc" && acc) {
        traces = [{ ...acc, name: "Acceleration (m/s²)" }];
        yTitle = "Acceleration (m/s²)";
    }

    Plotly.newPlot(graphDiv, traces, {
        ...(currentData.layout || {}),
        paper_bgcolor: "transparent",
        plot_bgcolor: "rgba(255,255,255,0.02)",
        font: { color: "#ffffff" },
        xaxis: { gridcolor: "rgba(255,255,255,0.05)", zeroline: false },
        yaxis: { title: yTitle, gridcolor: "rgba(255,255,255,0.05)", zeroline: false },
        margin: { t: 30, l: 45, r: 20, b: 45 }
    }, { responsive: true });
}

// =========================
// VIDEO PREVIEW
// =========================
videoInput?.addEventListener("change", () => {
    const file = videoInput.files?.[0];
    if (!file || !canvas) return;

    const video = document.createElement("video");
    const ctx = canvas.getContext("2d");

    video.src = URL.createObjectURL(file);

    video.addEventListener("loadeddata", () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        video.currentTime = 0.1;

        video.addEventListener("seeked", () => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        });
    });
});

// =========================
// CALIBRATION CLICK
// =========================
canvas?.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    selectedPoint = { x, y };

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "red";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();

    explanationDiv.innerHTML =
        `<p style="color:#00ffcc;">Calibration point selected ✔ (${x.toFixed(1)}, ${y.toFixed(1)})</p>`;
});

// =========================
// FORM SUBMIT
// =========================
form?.addEventListener("submit", async (e) => {
    e.preventDefault();

    loader?.classList.remove("hidden");
    graphDiv.innerHTML = "";
    explanationDiv.innerHTML = "";

    const formData = new FormData(form);

    if (selectedPoint) {
        formData.append("x", selectedPoint.x);
        formData.append("y", selectedPoint.y);
    }

    try {
        const res = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        loader?.classList.add("hidden");

        if (!res.ok || data.error) {
            explanationDiv.innerHTML =
                `<p style="color:#ff6b6b;">${data.error || "Analysis failed"}</p>`;
            return;
        }

        currentData = data;
        renderGraph("all");

        explanationDiv.innerHTML =
            `<div>${sanitizeHTML(data.explanation || "")}</div>`;

        if (data.physics_score) {
            const score = data.physics_score;

            explanationDiv.innerHTML += `
                <hr>
                <p><b>Physics Correctness Score:</b> ${score.confidence}%</p>
                <p><b>Score:</b> ${score.score.toFixed(2)} / 1.00</p>
            `;

            if (score.warnings?.length) {
                explanationDiv.innerHTML += `
                    <p style="color:#ffcc66;"><b>Warnings:</b></p>
                    <ul>
                        ${score.warnings.map(w => `<li>${w}</li>`).join("")}
                    </ul>
                `;
            }
        }

    } catch (err) {
        loader?.classList.add("hidden");
        console.error(err);
        explanationDiv.innerHTML =
            `<p style="color:#ff6b6b;">Server error</p>`;
    }
});

// =========================
// TAB SWITCHING
// =========================
tabs.forEach(tab => {
    tab.addEventListener("click", () => {
        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        renderGraph(tab.dataset.type);
    });
});

// =========================
// LAB REPORT GENERATOR
// =========================
reportBtn?.addEventListener("click", async () => {

    if (!currentData) {
        alert("Run analysis first bro.");
        return;
    }

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const expName = document.getElementById("experimentName")?.value || "Motion Experiment";

    let y = 10;

    const addSection = (title, content) => {
        if (y > 260) {
            doc.addPage();
            y = 10;
        }

        doc.setFont("helvetica", "bold");
        doc.text(title, 10, y);
        y += 6;

        doc.setFont("helvetica", "normal");

        const lines = doc.splitTextToSize(content, 180);

        lines.forEach(line => {
            if (y > 280) {
                doc.addPage();
                y = 10;
            }
            doc.text(line, 10, y);
            y += 6;
        });

        y += 4;
    };

    const explanation = currentData.explanation || "No explanation available.";

    const physicsScore = currentData.physics_score
        ? `Score: ${currentData.physics_score.score.toFixed(2)} / 1\nConfidence: ${currentData.physics_score.confidence}%`
        : "No score available.";

    // TITLE
    doc.setFontSize(18);
    doc.text(expName, 10, y);
    y += 10;

    doc.setFontSize(12);

    addSection("Aim", "To analyze motion using video-based tracking and derive position, velocity, and acceleration.");
    addSection("Observations", explanation);
    addSection("Graphs", "Position-Time, Velocity-Time, and Acceleration-Time graphs were generated.");
    addSection("Calculations", physicsScore);
    addSection("Conclusion", "The motion was successfully analyzed and validated using physics principles.");

    // WAIT to ensure graph is rendered
    await new Promise(res => setTimeout(res, 300));

    // ADD GRAPH IMAGE
    try {
        const imgData = await Plotly.toImage(graphDiv, {
            format: "png",
            width: 800,
            height: 500
        });

        doc.addPage();
        doc.setFontSize(14);
        doc.text("Graphs", 10, 10);

        doc.addImage(imgData, "PNG", 10, 20, 190, 120);

    } catch (err) {
        console.error("Graph export failed", err);
    }

    // TIMESTAMP
    const date = new Date().toLocaleString();
    doc.setFontSize(10);
    doc.text(`Generated on: ${date}`, 10, 290);

    doc.save("Motion_Report.pdf");
});
