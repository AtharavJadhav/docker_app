// frontend/src/App.js

import React, { useState } from "react";

function App() {
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [results, setResults] = useState("");

  const handleDeploy = async () => {
    try {
      const response = await fetch("http://localhost:5000/deploy/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // Update the request body to match the expected schema
        body: JSON.stringify({ models: selectedModels }),
      });

      if (!response.ok) {
        // If the response is not ok, log the status and the error message
        const errorData = await response.json();
        console.error(
          `HTTP error! status: ${response.status}`,
          errorData.detail
        );
      } else {
        // If the response is ok, log the success message
        const data = await response.json();
        console.log(data.message);
      }
    } catch (error) {
      // If there's a problem with the fetch operation, log the error
      console.error("There was a problem with the fetch operation:", error);
    }
  };

  const [isLoading, setIsLoading] = useState(false);

  const handleInference = async () => {
    setIsLoading(true); // Set loading to true when inference starts
    try {
      const response = await fetch("http://localhost:5000/infer/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_name: selectedModel }),
      });

      if (!response.ok) {
        // If the response is not ok, log the status and the error message
        const errorData = await response.json();
        console.error(
          `HTTP error! status: ${response.status}`,
          errorData.detail
        );
        setIsLoading(false); // Set loading to false if there is an error
      } else {
        // Polling mechanism to check if the inference is done
        const checkResults = async () => {
          const resultResponse = await fetch("http://localhost:5000/results/");
          if (!resultResponse.ok) {
            // If the response is not ok, log the status and the error message
            const errorData = await resultResponse.json();
            console.error(
              `HTTP error! status: ${resultResponse.status}`,
              errorData.detail
            );
            setIsLoading(false); // Set loading to false if there is an error
          } else {
            const resultData = await resultResponse.json();
            if (resultData.status !== "completed") {
              // If the results are not ready, wait for a second and check again
              setTimeout(checkResults, 1000);
            } else {
              // If the results are ready, set the results state and loading to false
              setResults(resultData.results);
              setIsLoading(false);
            }
          }
        };

        // Start the polling mechanism
        checkResults();
      }
    } catch (error) {
      // If there's a problem with the fetch operation, log the error
      console.error("There was a problem with the fetch operation:", error);
      setIsLoading(false); // Set loading to false if there is a fetch error
    }
  };

  return (
    <div>
      <div className="selector">
        <ModelDeploy
          setSelectedModels={setSelectedModels}
          handleDeploy={handleDeploy}
        />
        <ModelSelector setSelectedModel={setSelectedModel} />
        <Button onClick={handleInference}>Send Client Files</Button>
      </div>
      <div>
        {isLoading ? <p>Loading, please wait...</p> : null}
        {/* Rest of your component */}
      </div>
      <div className="results">
        <DisplayResult results={results} />
      </div>
    </div>
  );
}

function ModelSelector({ setSelectedModel }) {
  const handleChange = (event) => {
    setSelectedModel(event.target.value);
  };

  return (
    <div>
      <p>Choose a model:</p>
      <select onChange={handleChange}>
        <option value="mnist_model_onnx">MNIST ONNX</option>
        <option value="mnist_model_openvino">MNIST OpenVino</option>
        <option value="mnist_model_pytorch">MNIST Pytorch</option>
        <option value="bert_model_onnx">BERT ONNX</option>
      </select>
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

function Button({ onClick, children }) {
  return <button onClick={onClick}>{children}</button>;
}

function DisplayResult({ results }) {
  return (
    <div>
      <p>Results:</p>
      <pre>{results}</pre>
    </div>
  );
}

export default App;
