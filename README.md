# Depict 🎨

**AI-powered visual search for www.museumsofindia.gov.in**

*Inspired by [@ekzhang/dispict](https://github.com/ekzhang/dispict) - because great ideas deserve great implementations.*

---

## What is this?

Depict lets you search through Indian museum artworks using natural language. Think "painting of fruit" or "blue sculpture" and watch as AI finds visually similar pieces from museum collections across India.

The magic happens through CLIP (Contrastive Language-Image Pre-training) embeddings that understand both text and images, making it possible to find artworks that match your description even if the metadata doesn't contain those exact words.

## The Story Behind This

The original idea came from my friend [Karan Janthe](http://github.com/kmJ-007/) who thought it would be cool to make Indian museum collections more searchable. He helped a lot throughout this project, from brainstorming the initial concept to building a scrapper to get all the data and metadata of the images the trickiest parts.

We scraped data from [Museums of India](https://www.museumsofindia.gov.in/) and built this search engine. It was a fun learning experience that involved everything from web scraping to deploying ML models on Modal AI.

## Live Demo

<!-- 🔗 **[Try it here!](YOUR_DEPLOYED_URL)**    -->
Have not yet deployed because I used the free tier I was allocated this month. Will continue this next month and host it. But if anyone wants to try it locally, it does work.

**Note:** The live demo only includes embeddings for a subset of the data (ngma_blr.json & nat_del.json) due to compute limitations. The original plan was to generate embeddings for all museum records, but Google Colab doesn't handle this many images well, and Modal AI's free tier has its limits (we're students, after all 😅).

## What's Inside

### Data Sources
- **National Gallery of Modern Art, Bangalore** (ngma_blr.json)
- **National Museum, Delhi** (nat_del.json)
- **Nizam's Museum, Hyderabad** (nkm_hyd.json)
- **Indian Museum, Kolkata** (im_kol.json)
- **Goa State Museum** (gom_goa.json)
- **Allahabad Museum** (alh_ald.json)

*All scraped from the official Museums of India portal*

### Tech Stack
- **Backend:** Python + Modal AI (serverless deployment)
- **AI Model:** OpenAI's CLIP (ViT-B/32)
- **Frontend:** React.js
- **Data Processing:** h5py for embedding storage, numpy for similarity search
- **Styling:** Pure CSS (no frameworks, just vibes)

## How It Works

1. **Text Input:** You type something like "traditional painting"
2. **CLIP Magic:** The model converts your text into a 512-dimensional vector
3. **Similarity Search:** We compare this vector with pre-computed image embeddings
4. **Results:** You get artworks ranked by visual similarity

The embeddings are computed once and stored in HDF5 files for fast retrieval. Each search is essentially a cosine similarity calculation in high-dimensional space.

## Project Structure
```
bash
depict/
├── museum_modal.py           # Data processing & embedding generation
├── museum_search_api.py      # Modal API for search functionality
├── src/
│   ├── App.js               # Main React component
│   ├── components/
│   │   ├── SearchBox.js     # Search input interface
│   │   ├── ResultsGrid.js   # Results display grid
│   │   └── ArtworkCard.js   # Individual artwork cards
│   └── *.css               # Styling files
├── embeddings.h5           # Pre-computed CLIP embeddings
├── all_data.jsonl         # Combined museum data
└── README.md              # You are here!
```

## Running This Locally

### Prerequisites

- Node.js (for the frontend)
- Python 3.8+ (for backend/data processing)
- Modal AI account (free tier works for small datasets)

---

### Backend Setup

```bash
# Install Modal
pip install modal

# Deploy the search API
modal deploy museum_search_api.py

# Note: You'll need your own embeddings.h5 and all_data.jsonl files
```

### Frontend Setup

```bash
# Install dependencies
npm install

# Update the API endpoint in src/App.js with your Modal URL

# Start the dev server
npm start
```


### Full Dataset & Contributions

## Want the Full Dataset?

If you'd like us to process the complete museum collection (all 6 museums with thousands of artworks), feel free to send a little money our way to cover the GPU costs! Training CLIP embeddings for this many images requires more compute than what free tiers provide.

### Ways to contribute:

- Reach out to us on twttr/mail.  
- Or just star this repo if you think it's cool! ⭐

## Learnings

This project was a great deep dive into:

- Vector similarity search and how embeddings work in practice  
- Modal AI for serverless ML deployment (pretty neat platform!)  
- React state management for handling async API calls  
- Web scraping Indian government websites (surprisingly well-structured)  
- CLIP models and multimodal AI in general  

The most challenging part was probably getting the data pipeline right — from scraping inconsistent museum websites to handling failed image downloads during embedding generation.


## Limitations & Future Ideas

### Current limitations:

- Limited to text search (no image-to-image search yet)  
- Small dataset due to compute constraints  
- No detailed artwork metadata beyond basic fields  

### Cool future additions:

- Image upload for visual similarity search  
- More sophisticated filtering (by time period, medium, etc.)  
- Better mobile experience  
- Integration with museum APIs for real-time data

## Contributing

Found a bug? Have an idea? PRs welcome! This was built as a learning project, so there's definitely room for improvement.

---

## Credits

- **Original idea & major contributions:** [Karan Janthe](https://github.com/kmJ-007)  
- **Implementation & frontend:** [Dharmik](https://github.com/dhrm1k)  
- **Inspiration:** [@ekzhang/dispict](https://github.com/ekzhang/dispict)  
- **Data source:** [Museums of India](https://www.museumsofindia.gov.in/)  
- **AI Model:** OpenAI's CLIP  

_Built with ❤️ by Dharmik and Karan_

If you found this useful, consider starring the repo or sharing it with someone who might be interested in the intersection of AI and cultural heritage!
