import React, { useState } from 'react';
import './ArtworkCard.css';

// This component displays individual artwork information with clickable image
function ArtworkCard({ artwork, rank }) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  // Format similarity score as percentage
  const similarity = Math.round(artwork.similarity * 100);

  // Handle image click - opens the source image in a new tab
  const handleImageClick = () => {
    if (artwork.image_url) {
      window.open(artwork.image_url, '_blank');
    }
  };

  // Handle card click for more details (you can customize this)
  const handleCardClick = (e) => {
    // Don't trigger if clicking on the image (which has its own handler)
    if (e.target.closest('.artwork-image-container')) {
      return;
    }
    
    // You could add a modal with more details here
    console.log('Card clicked:', artwork);
  };

  return (
    <div className="artwork-card" onClick={handleCardClick}>
      {/* Similarity score badge */}
      <div className="similarity-badge">
        {similarity}% match
      </div>
      
      {/* Rank badge */}
      <div className="rank-badge">
        #{rank}
      </div>

      {/* Artwork image - clickable to view source */}
      <div 
        className="artwork-image-container"
        onClick={handleImageClick}
        title="Click to view full image"
      >
        {!imageLoaded && !imageError && (
          <div className="image-placeholder">
            <div className="loading-spinner-small"></div>
            <p>Loading image...</p>
          </div>
        )}
        
        {imageError ? (
          <div className="image-error">
            <span>🖼️</span>
            <p>Image not available</p>
          </div>
        ) : (
          <img
            src={artwork.image_url}
            alt={artwork.title}
            className={`artwork-image ${imageLoaded ? 'loaded' : ''}`}
            onLoad={() => setImageLoaded(true)}
            onError={() => setImageError(true)}
          />
        )}
        
        {/* Overlay that appears on hover */}
        <div className="image-overlay">
          <span className="view-icon">🔍</span>
          <p>View full image</p>
        </div>
      </div>

      {/* Artwork information */}
      <div className="artwork-info">
        
        <p className="artwork-artist">
          By {artwork.artist || 'Unknown Artist'}
        </p>
        
        <p className="artwork-museum">
          📍 {artwork.museum}
        </p>

        <div className="artwork-details">
          {artwork.object_type && (
            <span className="artwork-tag">{artwork.object_type}</span>
          )}

          {artwork.medium && (
            <span className="artwork-tag">{artwork.medium}</span>
          )}
        </div>

        {artwork.dimensions && (
          <p className="artwork-dimensions">
            📏 {artwork.dimensions}
          </p>
        )}

        {artwork.description && (
          <p className="artwork-description">
            {artwork.description.length > 120 
              ? artwork.description.substring(0, 120) + '...'
              : artwork.description
            }
          </p>
        )}
      </div>
    </div>
  );
}

export default ArtworkCard;