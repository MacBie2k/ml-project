import React, { useState } from "react";
import axios from 'axios';
import './App.css';

const App = () => {
    const [image, setImage] = useState(null);
    const [submitted, setSubmitted] = useState(false);
    const [predicted_index, setPredicted_index] = useState(-1);
    const categories = ["karton", "szkło", "metal", "papier", "plastik"];

    const handleImageUpload = (event) => {
        if (event.target.files && event.target.files[0]) {
            setImage(event.target.files[0]);
            setPredicted_index(-1)
            setSubmitted(false);
        }
    };

    const handleSubmit = async () => {
        if (image) {
            const formData = new FormData();
            formData.append('file', image);

            try {
                const response = await axios.post('http://localhost:5000/garbage', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                });
                const predictionIndex = response.data.predicted_index;
                setPredicted_index(predictionIndex)
            } catch (error) {
                console.error('Error uploading file:', error);
            }
        }
    };

    const getPredictionClassAndMessage = (index) => {
        const mappings = {
            0: { className: "paper-cardboard", message: `Ten śmieć to ${categories[0]}!`},
            1: { className: "glass", message: `Ten śmieć to ${categories[1]}!`},
            2: { className: "plastic-metal", message: `Ten śmieć to ${categories[2]}!`},
            3: { className: "paper-cardboard", message: `Ten śmieć to ${categories[3]}!`},
            4: { className: "plastic-metal", message: `Ten śmieć to ${categories[4]}!`}
        };

        return mappings[index] || { className: "", message: "" };
    }

    const { className, message } = getPredictionClassAndMessage(predicted_index);

    return (
        <div className="onepager">
            <h1 className="header">Śmieciosprawdzacz</h1>
            <br/>
            <div className="image-container">
                {image ? (
                    <img src={URL.createObjectURL(image)} alt="Wybrany obraz" />
                ) : (
                    <div>Proszę zaimportować obraz</div>
                )}
            </div>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            <button disabled={!image} onClick={handleSubmit}>Wyślij</button>
            {submitted && <div className="submission-success">Zdjęcie zostało pomyślnie przesłane!</div>}
            {predicted_index >= 0 && <div className={className}>{message}</div>}
        </div>
    );
};

export default App;
