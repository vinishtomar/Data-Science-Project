import React, { useState } from 'react';
import axios from 'axios';

function Forecast() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchPredictions = () => {
    setLoading(true);
    axios.post('http://localhost:5000/forecast', {
      data: [100, 120, 130, 150, 170],
      model: 'ARIMA',
      order: [1, 0, 1]
    })
    .then(response => {
      setPredictions(response.data.predictions);
      setLoading(false);
    })
    .catch(error => {
      console.error("Erreur de l'API:", error);
      setLoading(false);
    });
  };

  return (
    <div style={{
      maxWidth: '500px', 
      margin: 'auto', 
      padding: '20px', 
      boxShadow: '0 0 10px rgba(0,0,0,0.1)', 
      borderRadius: '10px'
    }}>
      <h2 style={{textAlign: 'center'}}>📈 Prévisions</h2>
      <button 
        onClick={fetchPredictions}
        style={{
          width: '100%', 
          padding: '10px', 
          marginBottom: '20px',
          cursor: 'pointer'
        }}
      >
        {loading ? "Chargement..." : "Obtenir les prévisions"}
      </button>

      {predictions.length > 0 && (
        <div>
          {predictions.map((value, index) => (
            <div key={index} style={{
              background: '#f0f4f8',
              padding: '10px',
              borderRadius: '5px',
              marginBottom: '10px'
            }}>
              <strong>Prévision {index + 1} :</strong> {value.toFixed(2)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Forecast;