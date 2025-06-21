import React, { useState } from 'react';
import SearchBox from './components/SearchBox';
import ResultsGrid from './components/ResultsGrid';
import './App.css';

// This is the main component that holds our entire application
function App() {
  // useState is a React Hook that lets us add state to our component
  // State is data that can change over time and affects what we display
  const [searchResults, setSearchResults] = useState([]); // Stores search results
  const [isLoading, setIsLoading] = useState(false);      // Tracks if we're searching
  const [error, setError] = useState(null);               // Stores any error messages
  const [searchQuery, setSearchQuery] = useState('');     // Current search query

  // This function handles the actual search
  const handleSearch = async (query) => {
    setIsLoading(true);  // Show loading state
    setError(null);      // Clear any previous errors
    setSearchQuery(query); // Store the search query
    
    try {
      // Prepare the request data for text search only
      const requestData = {
        query: query,
        top_k: 20  // Request 20 results
      };
      
      // Make API call to our Modal backend
      // Replace this URL with your actual Modal endpoint URL
      const response = await fetch('https://dhrm1k--museum-search.modal.run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setSearchResults([]);
      } else {
        setSearchResults(data.results || []);
      }
    } catch (err) {
      setError('Failed to search. Please try again.');
      setSearchResults([]);
    } finally {
      setIsLoading(false);  // Hide loading state
    }
  };

  // JSX is how we describe what the UI should look like
  return (
    <div className="App">
      <header className="App-header">
        <h1>Depict</h1>
        <p>Explore museumsofindia.gov.in collections with intelligent search</p>
      </header>
      
      <main>
        {/* SearchBox component handles user input */}
        <SearchBox onSearch={handleSearch} isLoading={isLoading} />
        
        {/* Show error message if there's an error */}
        {error && (
          <div className="error-message">
            <strong>Search Error:</strong> {error}
          </div>
        )}
        
        {/* ResultsGrid component displays the search results */}
        <ResultsGrid 
          results={searchResults} 
          isLoading={isLoading} 
          searchQuery={searchQuery}
        />
      </main>

              <div className="project-links">
          <a 
            href="https://github.com/dhrm1k/depict" 
            target="_blank" 
            rel="noopener noreferrer"
            className="github-link"
          >
             View Source
          </a>
          <span className="built-by">
            Built by{' '}
            <a 
              href="https://github.com/dhrm1k" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              Dharmik
            </a>
            {' & '}
            <a 
              href="https://github.com/kmJ-007" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              Karan
            </a>
          </span>
        </div>

    </div>
  );
}

export default App;