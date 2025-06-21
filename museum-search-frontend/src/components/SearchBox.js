import React, { useState } from 'react';
import './SearchBox.css';

// This component handles user input for text searching only
function SearchBox({ onSearch, isLoading }) {
  const [query, setQuery] = useState('');           // The text the user types

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault(); // Prevent page refresh
    if (query.trim()) { // Only search if there's actually text
      onSearch(query.trim(), 'text');
    }
  };

  return (
    <div className="search-box">
      <form onSubmit={handleSubmit}>
        <div className="search-header">
          <h2>Search</h2>
          <p>Discover artworks using AI-powered semantic search</p>
        </div>

        {/* Simplified input field for text search only */}
        <div className="search-input-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for art... (e.g., 'painting of fruit', 'blue sculpture', 'portrait')"
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !query.trim()}>
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {/* Example queries to help users */}
      <div className="example-queries">
        <p>Popular searches:</p>
        <div className="example-buttons">
          <button onClick={() => setQuery('painting of birds')}>painting of birds</button>
          <button onClick={() => setQuery('sculpture')}>sculpture</button>
          <button onClick={() => setQuery('portrait')}>portrait</button>
          <button onClick={() => setQuery('landscape')}>landscape</button>
          <button onClick={() => setQuery('blue artwork')}>blue artwork</button>
          <button onClick={() => setQuery('traditional art')}>traditional art</button>
        </div>
      </div>
    </div>
  );
}

export default SearchBox;