# 🚆 Indian Railway Visualization App

An interactive Streamlit-based web application to visualize Indian railway networks, stations, and routes using real data. This app lets users search for trains, plot real-time routes, explore stations geographically, and view train/station information in an intuitive and visually engaging way.

🌐 **Live Demo:** [Railway Visualization App](https://railway-visualization.streamlit.app/)


---

## 📌 Features

- 🔎 **Search by Train Number or Name**: Quickly find specific trains in the Indian railway database.
- 🗺️ **Interactive Route Maps**: Visualize train routes using `folium` and `pydeck` with dynamic zoom and markers.
- 📍 **Station Locator**: Clickable station points on a map with station metadata.
- 🧭 **Real-time Visualization**: Real-time plotting of train paths with start and end stations.
- 🧾 **Station and Route Details**: Display of latitude, longitude, state, and train route details.

---

## ⚙️ Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Visualization**: [Pydeck](https://deckgl.readthedocs.io/en/latest/), [Folium](https://python-visualization.github.io/folium/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/)
- **Data Sources**: Custom `routes.csv`, optional `stations.json` for metadata

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.7 or later
- pip (Python package installer)

### 📦 Installation

```bash
git clone https://github.com/your-username/railway-visualization.git
cd railway-visualization
pip install -r requirements.txt
```

### ▶️ Run the App Locally

```bash
streamlit run 975ee840-0d8c-469f-ab63-d39c23d5ede0.py
```

The app will open in your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🗃️ Sample Usage

- Enter train number (e.g., `12301`) or partial train name (e.g., `Rajdhani`) in the sidebar.
- View route plotted with stations on the map.
- Hover over markers to view station details.

---

## 📊 Data Format

### `routes.csv` Example

```
train_no,train_name,station_name,lat,lon
12301,Rajdhani Exp,New Delhi,28.6139,77.2090
12301,Rajdhani Exp,Kanpur Central,26.4499,80.3319
```

### `stations.json` Example

```json
{
  "NDLS": {
    "name": "New Delhi",
    "lat": 28.6139,
    "lon": 77.2090,
    "state": "Delhi"
  }
}
```

---

## 📚 Dependencies

Listed in `requirements.txt`:

```
streamlit
pandas
folium
pydeck
requests
```

To generate `requirements.txt` from your environment:

```bash
pip freeze > requirements.txt
```

---

## 🧑‍💻 Contribution Guide

1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push the changes:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request

---

## ❗ Known Issues

- Large CSVs may slow down rendering
- Incomplete station metadata can cause missing markers
- No real-time API integration (yet)

---

## 🔮 Future Improvements

- 🔄 Real-time train tracking using external APIs
- 🔍 Filter by region, zone, or state
- 🏎️ Performance optimization for larger datasets
- 🗺️ Advanced clustering for dense station maps

## 🙋‍♂️ Author

Developed by **Gautam Raj**  
Feel free to connect or contribute!

---

## ⭐ Give a Star!

If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!
