import React, { useState } from "react";
import './App.css'; 

const App = () => {
    const [image, setImage] = useState(null);
    const [submitted, setSubmitted] = useState(false);

    const handleImageUpload = (event) => {
        if (event.target.files && event.target.files[0]) {
            setImage(URL.createObjectURL(event.target.files[0]));
        }
    };

    const handleSubmit = () => {
        if (image) {
            setSubmitted(true);
        }
    };

    return (
        <div className="onepager">
            <h1 className="header">Śmieciosprawdzacz</h1>
            <br/>
            <div className="image-container">
                {image ? (
                    <img src={image} alt="Wybrany obraz" />
                ) : (
                    <div>Proszę zaimportować obraz</div>
                )}
            </div>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            <button disabled={!image} onClick={handleSubmit}>Wyślij</button>
            {submitted && <div className="submission-success">Zdjęcie zostało pomyślnie przesłane!</div>}
        </div>
    );
};

export default App;
