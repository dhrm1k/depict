import React from 'react';
import ArtworkCard from './ArtworkCard';
import './ResultsGrid.css';

// This component displays the search results in a grid layout
function ResultsGrid({ results, isLoading, searchQuery }) {
  
  // Show loading state
  if (isLoading) {
    return (
      <div className="results-container">
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Searching through museum collections...</p>
          {searchQuery && <p className="search-query">"{searchQuery}"</p>}
        </div>
      </div>
    );
  }

  // Show message when no results
  if (results.length === 0 && searchQuery) {
    return (
      <div className="results-container">
        <div className="no-results">
          <h3>No artworks found</h3>
          <p>No results for "<strong>{searchQuery}</strong>"</p>
          <p>Try different keywords or check spelling</p>
          <div className="search-tips">
            <h4>Search Tips:</h4>
            <ul>
              <li>Try broader terms like "painting" or "sculpture"</li>
              <li>Use descriptive words like "blue", "landscape", "portrait"</li>
              <li>Search for art styles like "traditional", "modern"</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  // Don't show anything if no search has been performed
  if (results.length === 0 && !searchQuery) {
    return null;
  }

  // Display results in a grid
  return (
    <div className="results-container">
      <div className="results-header">
        <h2>Found {results.length} artworks</h2>
        {searchQuery && (
          <p className="search-info">
            Results for: "<strong>{searchQuery}</strong>"
          </p>
        )}
        <p className="interaction-hint">
          💡 Click on any image to view it in full size
        </p>
      </div>
      
      <div className="results-grid">
        {/* Map over results and create an ArtworkCard for each */}
        {results.map((artwork, index) => (
          <ArtworkCard 
            key={artwork.id || index} 
            artwork={artwork} 
            rank={index + 1}
          />
        ))}
      </div>
    </div>
  );
}

export default ResultsGrid;