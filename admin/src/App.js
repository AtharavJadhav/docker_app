import React, { useState, useEffect } from "react";

function App() {
  const [selectedModels, setSelectedModels] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [dataShiftMetrics, setDataShiftMetrics] = useState({});

  const handleDeploy = async () => {
    try {
      const response = await fetch("http://localhost:5000/deploy/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ models: selectedModels }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error(
          `HTTP error! status: ${response.status}`,
          errorData.detail
        );
      } else {
        const data = await response.json();
        console.log(data.message);
      }
    } catch (error) {
      console.error("There was a problem with the fetch operation:", error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch("http://localhost:5000/metrics/");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error("Failed to fetch metrics:", error);
    }
  };

  const fetchDataShiftMetrics = async (modelName) => {
    try {
      const response = await fetch(`http://localhost:5000/compute_kl`, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ model_name: modelName }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setDataShiftMetrics((prev) => ({
        ...prev,
        [modelName]: data,
      }));
    } catch (error) {
      console.error(
        `Failed to fetch data shift metrics for ${modelName}:`,
        error
      );
    }
  };

  useEffect(() => {
    const intervalId = setInterval(fetchMetrics, 5000); // Fetch metrics every 5 seconds
    return () => clearInterval(intervalId); // Clean up interval on unmount
  }, []);

  return (
    <div>
      <ModelDeploy
        setSelectedModels={setSelectedModels}
        handleDeploy={handleDeploy}
      />
      <ModelMetrics
        metrics={metrics}
        dataShiftMetrics={dataShiftMetrics}
        fetchDataShiftMetrics={fetchDataShiftMetrics}
      />
    </div>
  );
}

function ModelDeploy({ setSelectedModels, handleDeploy }) {
  const handleCheckboxChange = (event) => {
    const value = event.target.value;
    setSelectedModels((prev) => {
      if (event.target.checked) {
        return [...prev, value];
      } else {
        return prev.filter((model) => model !== value);
      }
    });
  };

  return (
    <div>
      <p>Select models to deploy:</p>
      <input
        type="checkbox"
        id="mnist_model_onnx"
        name="mnist_model_onnx"
        value="mnist_model_onnx"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="mnist_model_onnx">MNIST ONNX</label>
      <br />
      <input
        type="checkbox"
        id="mnist_model_openvino"
        name="mnist_model_openvino"
        value="mnist_model_openvino"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="mnist_model_openvino">MNIST OpenVino</label>
      <br />
      <input
        type="checkbox"
        id="mnist_model_pytorch"
        name="mnist_model_pytorch"
        value="mnist_model_pytorch"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="mnist_model_pytorch">MNIST Pytorch</label>
      <br />
      <input
        type="checkbox"
        id="bert_model_onnx"
        name="bert_model_onnx"
        value="bert_model_onnx"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="bert_model_onnx">BERT ONNX</label>
      <br />
      <Button onClick={handleDeploy}>Deploy</Button>
    </div>
  );
}

function ModelMetrics({ metrics, dataShiftMetrics, fetchDataShiftMetrics }) {
  return (
    <div>
      <h2>Model Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Model Name</th>
            <th>Accuracy</th>
            <th>Latency (s)</th>
            <th>Throughput (req/s)</th>
            <th>Request Count</th>
            <th>KL Divergence</th>
            <th>Average Inference Histogram</th>
            <th>Retraining Needed</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys(metrics).map((modelName) => (
            <tr key={modelName}>
              <td>{modelName}</td>
              <td>{(metrics[modelName].accuracy * 100).toFixed(2)}%</td>
              <td>{metrics[modelName].latency.toFixed(4)}</td>
              <td>{metrics[modelName].throughput.toFixed(2)}</td>
              <td>{metrics[modelName].request_count}</td>
              <td>
                {dataShiftMetrics[modelName] &&
                dataShiftMetrics[modelName].kl_divergence !== null
                  ? dataShiftMetrics[modelName].kl_divergence.toFixed(4)
                  : "N/A"}
              </td>
              <td>
                {dataShiftMetrics[modelName] &&
                dataShiftMetrics[modelName].average_inference_histogram.length >
                  0
                  ? JSON.stringify(
                      dataShiftMetrics[modelName].average_inference_histogram
                    )
                  : "N/A"}
              </td>
              <td>
                {dataShiftMetrics[modelName]
                  ? dataShiftMetrics[modelName].retraining_needed
                  : "N/A"}
              </td>
              <td>
                <Button onClick={() => fetchDataShiftMetrics(modelName)}>
                  Calculate Data Shift
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Button({ onClick, children }) {
  return <button onClick={onClick}>{children}</button>;
}

export default App;
