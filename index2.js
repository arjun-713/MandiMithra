/**
 * MandiMithra - Navigation Bar Links Handler
 * This file manages the navbar interactions, API calls, and routing for the MandiMithra platform
 * Connects the frontend navigation elements with the Python backend
 */

// Configuration
const API_BASE_URL = "http://localhost:5000/api";
let currentUser = null;
let isLoggedIn = false;

// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all navigation elements
    initializeNavigation();
    setupMobileMenu();
    setupAuthButtons();
    
    // Check login status
    checkLoginStatus();
    
    // Setup language selection
    setupLanguageSelector();
});

/**
 * Initialize the main navigation links and handle active state
 */
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    
    // Setup event listeners for each nav link
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Only for links that should be handled by our SPA router
            if (this.getAttribute('data-spa-link') === 'true') {
                e.preventDefault();
                
                // Remove active class from all links
                navLinks.forEach(l => l.classList.remove('active'));
                
                // Add active class to clicked link
                this.classList.add('active');
                
                // Get the route from the href attribute
                const route = this.getAttribute('href').substring(1); // Remove the # from href
                
                // Handle navigation
                navigateTo(route);
            }
        });
    });
    
    // Handle initial route on page load
    handleInitialRoute();
}

/**
 * Check the current URL and activate the corresponding nav link
 */
function handleInitialRoute() {
    const path = window.location.pathname;
    const hash = window.location.hash.substring(1); // Remove the # from hash
    const route = hash || path.substring(1) || 'home'; // Default to home if no route
    
    // Find the corresponding link and activate it
    const activeLink = document.querySelector(`.nav-links a[href="#${route}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
        // Also navigate to that route to load the content
        navigateTo(route);
    } else {
        // Default to home if no matching link is found
        navigateTo('home');
    }
}

/**
 * Route to the appropriate view based on the route name
 * @param {string} route - The route to navigate to
 */
function navigateTo(route) {
    console.log(`Navigating to: ${route}`);
    
    // Update browser history
    window.history.pushState({ route }, route, `#${route}`);
    
    // Hide all views
    const views = document.querySelectorAll('.view');
    views.forEach(view => view.style.display = 'none');
    
    // Show the selected view
    const targetView = document.getElementById(`${route}-view`);
    if (targetView) {
        targetView.style.display = 'block';
        
        // Load any data needed for this view
        loadViewData(route);
    } else {
        console.error(`View not found: ${route}-view`);
        // Show 404 or redirect to home
        navigateTo('home');
    }
}

/**
 * Load data specific to each view
 * @param {string} route - The current route
 */
function loadViewData(route) {
    switch(route) {
        case 'home':
            // Load trending commodities for home page
            loadTrendingCommodities();
            break;
        case 'marketplace':
            // Load marketplace data
            loadMarketplaceData();
            break;
        case 'predictions':
            // Setup prediction form and load initial data
            setupPredictionForm();
            break;
        case 'profile':
            // Check if user is logged in
            if (!isLoggedIn) {
                navigateTo('login');
                return;
            }
            // Load user profile
            loadUserProfile();
            break;
        case 'insights':
            // Load insights form
            setupInsightsForm();
            break;
        // Add more cases as needed
    }
}

/**
 * Setup the hamburger menu for mobile view
 */
function setupMobileMenu() {
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', function() {
            navLinks.classList.toggle('show');
            this.classList.toggle('active');
        });
    }
}

/**
 * Setup authentication buttons (login/logout)
 */
function setupAuthButtons() {
    const loginBtn = document.querySelector('.btn-login');
    const signupBtn = document.querySelector('.btn-signup');
    
    if (loginBtn) {
        loginBtn.addEventListener('click', function() {
            if (isLoggedIn) {
                // Handle logout
                logoutUser();
            } else {
                // Navigate to login page
                navigateTo('login');
            }
        });
    }
    
    if (signupBtn) {
        signupBtn.addEventListener('click', function() {
            navigateTo('signup');
        });
    }
}

/**
 * Check if user is logged in
 */
function checkLoginStatus() {
    // Check localStorage for authentication token
    const token = localStorage.getItem('authToken');
    
    if (token) {
        // Verify token with backend
        fetch(`${API_BASE_URL}/verify-token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            }
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                // Token invalid, clear it
                localStorage.removeItem('authToken');
                updateUIForLoggedOut();
                throw new Error('Invalid token');
            }
        })
        .then(data => {
            if (data.valid) {
                currentUser = data.user;
                isLoggedIn = true;
                updateUIForLoggedIn(data.user);
            } else {
                updateUIForLoggedOut();
            }
        })
        .catch(error => {
            console.error('Error verifying token:', error);
            updateUIForLoggedOut();
        });
    } else {
        updateUIForLoggedOut();
    }
}

/**
 * Update UI for logged in user
 * @param {Object} user - User information
 */
function updateUIForLoggedIn(user) {
    const loginBtn = document.querySelector('.btn-login');
    const signupBtn = document.querySelector('.btn-signup');
    const userMenu = document.querySelector('.user-menu');
    
    if (loginBtn) loginBtn.textContent = 'Logout';
    if (signupBtn) signupBtn.style.display = 'none';
    
    // Show user menu if it exists
    if (userMenu) {
        userMenu.style.display = 'flex';
        
        // Update user name
        const userName = userMenu.querySelector('.user-name');
        if (userName && user.name) {
            userName.textContent = user.name;
        }
    }
    
    // Add profile link to nav if not exists
    const navLinks = document.querySelector('.nav-links');
    if (navLinks && !navLinks.querySelector('a[href="#profile"]')) {
        const profileLi = document.createElement('li');
        const profileLink = document.createElement('a');
        profileLink.href = '#profile';
        profileLink.textContent = 'My Profile';
        profileLink.setAttribute('data-spa-link', 'true');
        profileLink.addEventListener('click', function(e) {
            e.preventDefault();
            navigateTo('profile');
        });
        profileLi.appendChild(profileLink);
        navLinks.appendChild(profileLi);
    }
}

/**
 * Update UI for logged out state
 */
function updateUIForLoggedOut() {
    const loginBtn = document.querySelector('.btn-login');
    const signupBtn = document.querySelector('.btn-signup');
    const userMenu = document.querySelector('.user-menu');
    
    if (loginBtn) loginBtn.textContent = 'Login';
    if (signupBtn) signupBtn.style.display = 'inline-block';
    
    // Hide user menu
    if (userMenu) userMenu.style.display = 'none';
    
    // Remove profile link from nav
    const profileLink = document.querySelector('.nav-links a[href="#profile"]');
    if (profileLink) {
        const profileLi = profileLink.parentElement;
        profileLi.parentElement.removeChild(profileLi);
    }
    
    // Reset global state
    currentUser = null;
    isLoggedIn = false;
}

/**
 * Log out current user
 */
function logoutUser() {
    // Clear authentication token
    localStorage.removeItem('authToken');
    
    // Update UI
    updateUIForLoggedOut();
    
    // Navigate to home
    navigateTo('home');
    
    // Show logout message
    showNotification('You have been logged out', 'success');
}

/**
 * Setup language selector
 */
function setupLanguageSelector() {
    const languageSelector = document.querySelector('.language-selector');
    
    if (languageSelector) {
        languageSelector.addEventListener('change', function() {
            const selectedLanguage = this.value;
            changeLanguage(selectedLanguage);
        });
    }
}

/**
 * Change the application language
 * @param {string} language - Language code
 */
function changeLanguage(language) {
    // Store language preference
    localStorage.setItem('language', language);
    
    // In a real app, you would load language strings and update UI
    console.log(`Changed language to: ${language}`);
    
    // TODO: Implement actual language change
    // This would typically involve loading a new set of strings
    // and updating all text elements on the page
}

/**
 * Show notification message
 * @param {string} message - Message to display
 * @param {string} type - Message type (success, error, info)
 */
function showNotification(message, type = 'info') {
    // Check if notification container exists, create if not
    let notificationContainer = document.querySelector('.notification-container');
    
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.className = 'notification-container';
        document.body.appendChild(notificationContainer);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add close button
    const closeBtn = document.createElement('span');
    closeBtn.className = 'notification-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.onclick = function() {
        notification.remove();
    };
    notification.appendChild(closeBtn);
    
    // Add to container
    notificationContainer.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

/**
 * Load trending commodities for home page
 */
function loadTrendingCommodities() {
    // Show loading state
    const trendingContainer = document.querySelector('.trending-commodities');
    if (trendingContainer) {
        trendingContainer.innerHTML = '<div class="loading">Loading trending commodities...</div>';
        
        // Fetch trending commodities from API
        fetch(`${API_BASE_URL}/trending`)
            .then(response => {
                if (response.ok) return response.json();
                throw new Error('Failed to load trending commodities');
            })
            .then(data => {
                // Process data and update UI
                // TODO: Implement trending commodities display
            })
            .catch(error => {
                console.error('Error loading trending commodities:', error);
                trendingContainer.innerHTML = '<div class="error">Failed to load trending commodities</div>';
            });
    }
}

// TODO: Implement remaining data loading functions for different views
// - loadMarketplaceData()
// - setupPredictionForm()
// - loadUserProfile()
// - setupInsightsForm()

// Event listener for back/forward browser buttons
window.addEventListener('popstate', function(event) {
    if (event.state && event.state.route) {
        navigateTo(event.state.route);
    } else {
        handleInitialRoute();
    }
});
