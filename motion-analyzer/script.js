const form = document.getElementById("uploadForm");
const graphDiv = document.getElementById("graph");
const explanationDiv = document.getElementById("explanation");
const loader = document.getElementById("loader");

let currentData = null;

// =========================
// RENDER GRAPH FUNCTION
// =========================
function renderGraph(type = "all") {
    if (!currentData) return;

    let traces = [];

    if (type === "all") {
        traces = currentData.data;
    } else if (type === "pos") {
        traces = [currentData.data[0]];
    } else if (type === "vel") {
        traces = [currentData.data[1]];
    } else if (type === "acc") {
        traces = [currentData.data[2]];
    }

    Plotly.newPlot(graphDiv, traces, {
        ...currentData.layout,

        paper_bgcolor: "transparent",
        plot_bgcolor: "rgba(255,255,255,0.02)",
        font: { color: "#ffffff" },

        xaxis: {
            gridcolor: "rgba(255,255,255,0.05)",
            zeroline: false
        },
        yaxis: {
            gridcolor: "rgba(255,255,255,0.05)",
            zeroline: false
        },

        margin: { t: 30, l: 40, r: 20, b: 40 }
    }, { responsive: true });
}

// =========================
// FORM SUBMIT
// =========================
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    loader.classList.remove("hidden");
    graphDiv.innerHTML = "";
    explanationDiv.innerHTML = "";

    const formData = new FormData(form);

    try {
        const res = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        loader.classList.add("hidden");

        if (data.error) {
            explanationDiv.innerHTML = `<p style="color:#ff6b6b;">${data.error}</p>`;
            return;
        }

        // store data globally
        currentData = data;

        // default render
        renderGraph("all");

        explanationDiv.innerHTML = `<p>${data.explanation}</p>`;

    } catch (err) {
        loader.classList.add("hidden");
        explanationDiv.innerHTML = `<p style="color:#ff6b6b;">Server error</p>`;
        console.error(err);
    }
});

// =========================
// TAB SWITCHING
// =========================
const tabs = document.querySelectorAll(".tab");

tabs.forEach(tab => {
    tab.addEventListener("click", () => {

        // remove active
        tabs.forEach(t => t.classList.remove("active"));

        // set active
        tab.classList.add("active");

        // render selected graph
        const type = tab.dataset.type;
        renderGraph(type);
    });
});