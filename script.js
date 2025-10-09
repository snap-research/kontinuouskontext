// Basic JavaScript functionality for the website

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.toc a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading animation for images
    const images = document.querySelectorAll('img');
    
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });
        
        img.addEventListener('error', function() {
            // Handle missing images gracefully
            this.style.display = 'block';
            this.style.backgroundColor = '#f8f9fa';
            this.style.border = '2px dashed #ddd';
            this.style.minHeight = '200px';
            this.style.position = 'relative';
            
            // Add placeholder text
            const placeholder = document.createElement('div');
            placeholder.textContent = 'Image placeholder - ' + this.alt;
            placeholder.style.position = 'absolute';
            placeholder.style.top = '50%';
            placeholder.style.left = '50%';
            placeholder.style.transform = 'translate(-50%, -50%)';
            placeholder.style.color = '#999';
            placeholder.style.textAlign = 'center';
            
            this.parentNode.style.position = 'relative';
            this.parentNode.appendChild(placeholder);
        });
    });

    // Add copy functionality to citation
    const citationBox = document.querySelector('.citation-box pre');
    if (citationBox) {
        citationBox.addEventListener('click', function() {
            const text = this.textContent;
            navigator.clipboard.writeText(text).then(function() {
                // Show temporary feedback
                const originalText = citationBox.textContent;
                citationBox.style.backgroundColor = '#27ae60';
                setTimeout(() => {
                    citationBox.style.backgroundColor = '#2c3e50';
                }, 1000);
            });
        });
        
        // Add click hint
        citationBox.style.cursor = 'pointer';
        citationBox.title = 'Click to copy citation';
    }

    // Add fade-in animation for sections
    const sections = document.querySelectorAll('.section');
    
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
});

// Function to update the page title and content dynamically
function updateProjectInfo(projectData) {
    if (projectData.title) {
        document.querySelector('.title').textContent = projectData.title;
        document.title = projectData.title;
    }
    
    if (projectData.authors) {
        const authorsContainer = document.querySelector('.authors');
        authorsContainer.innerHTML = '';
        projectData.authors.forEach(author => {
            const span = document.createElement('span');
            span.className = 'author';
            span.textContent = author;
            authorsContainer.appendChild(span);
        });
    }
    
    if (projectData.affiliation) {
        document.querySelector('.affiliation').textContent = projectData.affiliation;
    }
    
    if (projectData.date) {
        document.querySelector('.date span').textContent = projectData.date;
    }
    
    if (projectData.abstract) {
        document.querySelector('.abstract').innerHTML = projectData.abstract;
    }
}

// Example usage:
// updateProjectInfo({
//     title: "My Amazing Project: Revolutionary Approach to AI",
//     authors: ["John Doe", "Jane Smith", "Bob Johnson"],
//     affiliation: "University of Technology",
//     date: "December 2024",
//     abstract: "We present <strong>My Amazing Project</strong>, a novel approach..."
// });

// Teaser video with delayed restart
document.addEventListener('DOMContentLoaded', function() {
    const teaserVideo = document.getElementById('teaser-video');
    if (teaserVideo) {
        teaserVideo.addEventListener('ended', function() {
            setTimeout(() => {
                teaserVideo.currentTime = 0;
                teaserVideo.play();
            }, 2000); // 2 second delay
        });
    }
});

// Image Edit Slider Functionality
document.addEventListener('DOMContentLoaded', function() {
    const strengthSlider = document.getElementById('strength-slider');
    const resultImage = document.getElementById('result-image');
    
    // Define the mapping of slider values to pre-saved images from aesthetic_model2_teaser_pixar folder
    const imageMapping = {
        0: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_0.png',
            caption: 'Original image (0.0 edit strength)'
        },
        8: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_1.png',
            caption: 'Light editing effect (0.08 edit strength)'
        },
        17: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_2.png',
            caption: 'Light editing effect (0.17 edit strength)'
        },
        25: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_3.png',
            caption: 'Moderate editing effect (0.25 edit strength)'
        },
        33: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_4.png',
            caption: 'Moderate editing effect (0.33 edit strength)'
        },
        42: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_5.png',
            caption: 'Balanced editing effect (0.42 edit strength)'
        },
        50: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_6.png',
            caption: 'Balanced editing effect (0.5 edit strength)'
        },
        58: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_7.png',
            caption: 'Strong editing effect (0.58 edit strength)'
        },
        67: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_8.png',
            caption: 'Strong editing effect (0.67 edit strength)'
        },
        75: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_9.png',
            caption: 'Very strong editing effect (0.75 edit strength)'
        },
        83: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_10.png',
            caption: 'Very strong editing effect (0.83 edit strength)'
        },
        92: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_11.png',
            caption: 'Near maximum editing effect (0.92 edit strength)'
        },
        100: {
            src: 'assets/aesthetic_model2_teaser_pixar/image_11.png',
            caption: 'Maximum editing effect (1.0 edit strength)'
        }
    };
    
    if (strengthSlider && resultImage) {
        // Update display and image when slider changes
        strengthSlider.addEventListener('input', function() {
            const sliderValue = parseInt(this.value);
            
            // Find the closest available image
            const availableValues = Object.keys(imageMapping).map(Number).sort((a, b) => a - b);
            let closestValue = availableValues[0];
            
            for (let i = 0; i < availableValues.length; i++) {
                if (Math.abs(availableValues[i] - sliderValue) < Math.abs(closestValue - sliderValue)) {
                    closestValue = availableValues[i];
                }
            }
            
            const imageData = imageMapping[closestValue];
            
            if (imageData) {
                // Add subtle fade effect during image change
                resultImage.style.opacity = '0.8';
                
                // Change image source
                resultImage.src = imageData.src;
                
                // Handle image load to restore opacity
                resultImage.onload = function() {
                    this.style.opacity = '1';
                };
                
                // Handle image error (fallback to first teaser image)
                resultImage.onerror = function() {
                    this.src = 'assets/teaser/row_00_col_00.png';
                    this.style.opacity = '1';
                };
            }
        });
        
        // Initialize with the default value
        const initialValue = parseInt(strengthSlider.value);
        
        // Set initial image
        const initialImageData = imageMapping[initialValue];
        if (initialImageData) {
            resultImage.src = initialImageData.src;
        }
    }
});

// Function to add custom image mappings (for easy customization)
function updateImageMapping(customMapping) {
    // This allows you to easily update the image mappings
    // Example usage:
    // updateImageMapping({
    //     0: { src: 'assets/my_original.png', caption: 'My original image' },
    //     50: { src: 'assets/my_edit_50.png', caption: 'My 50% edit' },
    //     100: { src: 'assets/my_edit_100.png', caption: 'My full edit' }
    // });
    
    Object.assign(imageMapping, customMapping);
}

// Function to preload all images for smooth transitions
function preloadImages() {
    const imagePaths = [
        'assets/aesthetic_model2_teaser_pixar/image_0.png',
        'assets/aesthetic_model2_teaser_pixar/image_1.png',
        'assets/aesthetic_model2_teaser_pixar/image_2.png',
        'assets/aesthetic_model2_teaser_pixar/image_3.png',
        'assets/aesthetic_model2_teaser_pixar/image_4.png',
        'assets/aesthetic_model2_teaser_pixar/image_5.png',
        'assets/aesthetic_model2_teaser_pixar/image_6.png',
        'assets/aesthetic_model2_teaser_pixar/image_7.png',
        'assets/aesthetic_model2_teaser_pixar/image_8.png',
        'assets/aesthetic_model2_teaser_pixar/image_9.png',
        'assets/aesthetic_model2_teaser_pixar/image_10.png',
        'assets/aesthetic_model2_teaser_pixar/image_11.png'
    ];
    
    imagePaths.forEach(src => {
        const img = new Image();
        img.src = src;
    });
}

// Preload images when page loads
document.addEventListener('DOMContentLoaded', preloadImages);
document.addEventListener('DOMContentLoaded', preloadStrengthControlledImages);
document.addEventListener('DOMContentLoaded', preloadExampleImages);
document.addEventListener('DOMContentLoaded', preloadResultImages);

// Function to preload strength controlled images for smooth transitions
function preloadStrengthControlledImages() {
    const strengthControlledImagePaths = [
        // Horse uncle images (12 images)
        'assets/horse_uncle/image_0.png',
        'assets/horse_uncle/image_1.png',
        'assets/horse_uncle/image_2.png',
        'assets/horse_uncle/image_3.png',
        'assets/horse_uncle/image_4.png',
        'assets/horse_uncle/image_5.png',
        'assets/horse_uncle/image_6.png',
        'assets/horse_uncle/image_7.png',
        'assets/horse_uncle/image_8.png',
        'assets/horse_uncle/image_9.png',
        'assets/horse_uncle/image_10.png',
        'assets/horse_uncle/image_11.png',
        // Car reduce size2 images (12 images)
        'assets/car_reduce_size2/image_0.png',
        'assets/car_reduce_size2/image_1.png',
        'assets/car_reduce_size2/image_2.png',
        'assets/car_reduce_size2/image_3.png',
        'assets/car_reduce_size2/image_4.png',
        'assets/car_reduce_size2/image_5.png',
        'assets/car_reduce_size2/image_6.png',
        'assets/car_reduce_size2/image_7.png',
        'assets/car_reduce_size2/image_8.png',
        'assets/car_reduce_size2/image_9.png',
        'assets/car_reduce_size2/image_10.png',
        'assets/car_reduce_size2/image_11.png',
        // Person blur red hairs images (12 images)
        'assets/person_blur_red_hairs/image_0.png',
        'assets/person_blur_red_hairs/image_1.png',
        'assets/person_blur_red_hairs/image_2.png',
        'assets/person_blur_red_hairs/image_3.png',
        'assets/person_blur_red_hairs/image_4.png',
        'assets/person_blur_red_hairs/image_5.png',
        'assets/person_blur_red_hairs/image_6.png',
        'assets/person_blur_red_hairs/image_7.png',
        'assets/person_blur_red_hairs/image_8.png',
        'assets/person_blur_red_hairs/image_9.png',
        'assets/person_blur_red_hairs/image_10.png',
        'assets/person_blur_red_hairs/image_11.png'
    ];
    
    strengthControlledImagePaths.forEach(path => {
        const img = new Image();
        img.src = path;
    });
}

// Function to preload example images for smooth transitions
function preloadExampleImages() {
    const exampleImagePaths = [
        // Jacket Leather images (12 images)
        'assets/jacket_leather/image_0.png',
        'assets/jacket_leather/image_1.png',
        'assets/jacket_leather/image_2.png',
        'assets/jacket_leather/image_3.png',
        'assets/jacket_leather/image_4.png',
        'assets/jacket_leather/image_5.png',
        'assets/jacket_leather/image_6.png',
        'assets/jacket_leather/image_7.png',
        'assets/jacket_leather/image_8.png',
        'assets/jacket_leather/image_9.png',
        'assets/jacket_leather/image_10.png',
        'assets/jacket_leather/image_11.png',
        // Glasses Aviator images (12 images)
        'assets/glasses_aviator/image_0.png',
        'assets/glasses_aviator/image_1.png',
        'assets/glasses_aviator/image_2.png',
        'assets/glasses_aviator/image_3.png',
        'assets/glasses_aviator/image_4.png',
        'assets/glasses_aviator/image_5.png',
        'assets/glasses_aviator/image_6.png',
        'assets/glasses_aviator/image_7.png',
        'assets/glasses_aviator/image_8.png',
        'assets/glasses_aviator/image_9.png',
        'assets/glasses_aviator/image_10.png',
        'assets/glasses_aviator/image_11.png',
        // Lamp Yellow images (11 images)
        'assets/lamp_yellow/image_0.png',
        'assets/lamp_yellow/image_1.png',
        'assets/lamp_yellow/image_2.png',
        'assets/lamp_yellow/image_3.png',
        'assets/lamp_yellow/image_4.png',
        'assets/lamp_yellow/image_5.png',
        'assets/lamp_yellow/image_6.png',
        'assets/lamp_yellow/image_7.png',
        'assets/lamp_yellow/image_8.png',
        'assets/lamp_yellow/image_9.png',
        'assets/lamp_yellow/image_10.png',
        // Man Fur Jacket Bike images (9 images)
        'assets/man_fur_jacket_bike/image_0.png',
        'assets/man_fur_jacket_bike/image_1.png',
        'assets/man_fur_jacket_bike/image_2.png',
        'assets/man_fur_jacket_bike/image_3.png',
        'assets/man_fur_jacket_bike/image_4.png',
        'assets/man_fur_jacket_bike/image_5.png',
        'assets/man_fur_jacket_bike/image_6.png',
        'assets/man_fur_jacket_bike/image_7.png',
        'assets/man_fur_jacket_bike/image_8.png',
        // Model2 Sunlight images (12 images)
        'assets/model2_sunlight/image_0.png',
        'assets/model2_sunlight/image_1.png',
        'assets/model2_sunlight/image_2.png',
        'assets/model2_sunlight/image_3.png',
        'assets/model2_sunlight/image_4.png',
        'assets/model2_sunlight/image_5.png',
        'assets/model2_sunlight/image_6.png',
        'assets/model2_sunlight/image_7.png',
        'assets/model2_sunlight/image_8.png',
        'assets/model2_sunlight/image_9.png',
        'assets/model2_sunlight/image_10.png',
        'assets/model2_sunlight/image_11.png',
        // Panda Indoor2 Husky Dog images (12 images)
        'assets/panda_indoor2_husky_dog/image_0.png',
        'assets/panda_indoor2_husky_dog/image_1.png',
        'assets/panda_indoor2_husky_dog/image_2.png',
        'assets/panda_indoor2_husky_dog/image_3.png',
        'assets/panda_indoor2_husky_dog/image_4.png',
        'assets/panda_indoor2_husky_dog/image_5.png',
        'assets/panda_indoor2_husky_dog/image_6.png',
        'assets/panda_indoor2_husky_dog/image_7.png',
        'assets/panda_indoor2_husky_dog/image_8.png',
        'assets/panda_indoor2_husky_dog/image_9.png',
        'assets/panda_indoor2_husky_dog/image_10.png',
        'assets/panda_indoor2_husky_dog/image_11.png',
        // Person Blur Pixar images (12 images)
        'assets/person_blur_pixar/image_0.png',
        'assets/person_blur_pixar/image_1.png',
        'assets/person_blur_pixar/image_2.png',
        'assets/person_blur_pixar/image_3.png',
        'assets/person_blur_pixar/image_4.png',
        'assets/person_blur_pixar/image_5.png',
        'assets/person_blur_pixar/image_6.png',
        'assets/person_blur_pixar/image_7.png',
        'assets/person_blur_pixar/image_8.png',
        'assets/person_blur_pixar/image_9.png',
        'assets/person_blur_pixar/image_10.png',
        'assets/person_blur_pixar/image_11.png',
        // Tibbet Autumn images (12 images)
        'assets/tibbet_autumn/image_0.png',
        'assets/tibbet_autumn/image_1.png',
        'assets/tibbet_autumn/image_2.png',
        'assets/tibbet_autumn/image_3.png',
        'assets/tibbet_autumn/image_4.png',
        'assets/tibbet_autumn/image_5.png',
        'assets/tibbet_autumn/image_6.png',
        'assets/tibbet_autumn/image_7.png',
        'assets/tibbet_autumn/image_8.png',
        'assets/tibbet_autumn/image_9.png',
        'assets/tibbet_autumn/image_10.png',
        'assets/tibbet_autumn/image_11.png',
        // Venice1 Grow Vegetation images (12 images)
        'assets/venice1_Grow_vegetation_on_t_3/image_0.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_1.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_2.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_3.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_4.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_5.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_6.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_7.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_8.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_9.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_10.png',
        'assets/venice1_Grow_vegetation_on_t_3/image_11.png',
        // Enfield3 Winter Snow images (12 images)
        'assets/enfield3_winter_snow/image_0.png',
        'assets/enfield3_winter_snow/image_1.png',
        'assets/enfield3_winter_snow/image_2.png',
        'assets/enfield3_winter_snow/image_3.png',
        'assets/enfield3_winter_snow/image_4.png',
        'assets/enfield3_winter_snow/image_5.png',
        'assets/enfield3_winter_snow/image_6.png',
        'assets/enfield3_winter_snow/image_7.png',
        'assets/enfield3_winter_snow/image_8.png',
        'assets/enfield3_winter_snow/image_9.png',
        'assets/enfield3_winter_snow/image_10.png',
        'assets/enfield3_winter_snow/image_11.png'
    ];
    
    exampleImagePaths.forEach(src => {
        const img = new Image();
        img.src = src;
    });
}

// Function to preload result images for smooth transitions
function preloadResultImages() {
    const resultImagePaths = [
        // Aesthetic Model 3 images (12 images)
        'assets/aesthetic_model3/image_0.png',
        'assets/aesthetic_model3/image_1.png',
        'assets/aesthetic_model3/image_2.png',
        'assets/aesthetic_model3/image_3.png',
        'assets/aesthetic_model3/image_4.png',
        'assets/aesthetic_model3/image_5.png',
        'assets/aesthetic_model3/image_6.png',
        'assets/aesthetic_model3/image_7.png',
        'assets/aesthetic_model3/image_8.png',
        'assets/aesthetic_model3/image_9.png',
        'assets/aesthetic_model3/image_10.png',
        'assets/aesthetic_model3/image_11.png',
        // Horse Uncle images (12 images) - already loaded above
        // Man Jacket images (12 images) - already loaded above
        // Person Blur images (12 images) - already loaded above
        // Glasses images (12 images)
        'assets/glasses_img4/image_0.png',
        'assets/glasses_img4/image_1.png',
        'assets/glasses_img4/image_2.png',
        'assets/glasses_img4/image_3.png',
        'assets/glasses_img4/image_4.png',
        'assets/glasses_img4/image_5.png',
        'assets/glasses_img4/image_6.png',
        'assets/glasses_img4/image_7.png',
        'assets/glasses_img4/image_8.png',
        'assets/glasses_img4/image_9.png',
        'assets/glasses_img4/image_10.png',
        'assets/glasses_img4/image_11.png',
        // Panda images (12 images)
        'assets/panda/image_0.png',
        'assets/panda/image_1.png',
        'assets/panda/image_2.png',
        'assets/panda/image_3.png',
        'assets/panda/image_4.png',
        'assets/panda/image_5.png',
        'assets/panda/image_6.png',
        'assets/panda/image_7.png',
        'assets/panda/image_8.png',
        'assets/panda/image_9.png',
        'assets/panda/image_10.png',
        'assets/panda/image_11.png'
    ];
    
    resultImagePaths.forEach(src => {
        const img = new Image();
        img.src = src;
    });
}

// Results Section Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Define image mappings for each result
    const resultMappings = {
        result1: {
            0: 'assets/aesthetic_model3/image_0.png',
            8: 'assets/aesthetic_model3/image_1.png',
            17: 'assets/aesthetic_model3/image_2.png',
            25: 'assets/aesthetic_model3/image_3.png',
            33: 'assets/aesthetic_model3/image_4.png',
            42: 'assets/aesthetic_model3/image_5.png',
            50: 'assets/aesthetic_model3/image_6.png',
            58: 'assets/aesthetic_model3/image_7.png',
            67: 'assets/aesthetic_model3/image_8.png',
            75: 'assets/aesthetic_model3/image_9.png',
            83: 'assets/aesthetic_model3/image_10.png',
            92: 'assets/aesthetic_model3/image_11.png',
            100: 'assets/aesthetic_model3/image_11.png'
        },
        result2: {
            0: 'assets/horse_uncle/image_0.png',
            8: 'assets/horse_uncle/image_1.png',
            17: 'assets/horse_uncle/image_2.png',
            25: 'assets/horse_uncle/image_3.png',
            33: 'assets/horse_uncle/image_4.png',
            42: 'assets/horse_uncle/image_5.png',
            50: 'assets/horse_uncle/image_6.png',
            58: 'assets/horse_uncle/image_7.png',
            67: 'assets/horse_uncle/image_8.png',
            75: 'assets/horse_uncle/image_9.png',
            83: 'assets/horse_uncle/image_10.png',
            92: 'assets/horse_uncle/image_11.png',
            100: 'assets/horse_uncle/image_11.png'
        },
        result3: {
            0: 'assets/man_jacket/image_0.png',
            8: 'assets/man_jacket/image_1.png',
            17: 'assets/man_jacket/image_2.png',
            25: 'assets/man_jacket/image_3.png',
            33: 'assets/man_jacket/image_4.png',
            42: 'assets/man_jacket/image_5.png',
            50: 'assets/man_jacket/image_6.png',
            58: 'assets/man_jacket/image_7.png',
            67: 'assets/man_jacket/image_8.png',
            75: 'assets/man_jacket/image_9.png',
            83: 'assets/man_jacket/image_10.png',
            92: 'assets/man_jacket/image_11.png',
            100: 'assets/man_jacket/image_11.png'
        },
        result4: {
            0: 'assets/person_blur/image_0.png',
            8: 'assets/person_blur/image_1.png',
            17: 'assets/person_blur/image_2.png',
            25: 'assets/person_blur/image_3.png',
            33: 'assets/person_blur/image_4.png',
            42: 'assets/person_blur/image_5.png',
            50: 'assets/person_blur/image_6.png',
            58: 'assets/person_blur/image_7.png',
            67: 'assets/person_blur/image_8.png',
            75: 'assets/person_blur/image_9.png',
            83: 'assets/person_blur/image_10.png',
            92: 'assets/person_blur/image_11.png',
            100: 'assets/person_blur/image_11.png'
        },
        result5: {
            0: 'assets/glasses_img4/image_0.png',
            8: 'assets/glasses_img4/image_1.png',
            17: 'assets/glasses_img4/image_2.png',
            25: 'assets/glasses_img4/image_3.png',
            33: 'assets/glasses_img4/image_4.png',
            42: 'assets/glasses_img4/image_5.png',
            50: 'assets/glasses_img4/image_6.png',
            58: 'assets/glasses_img4/image_7.png',
            67: 'assets/glasses_img4/image_8.png',
            75: 'assets/glasses_img4/image_9.png',
            83: 'assets/glasses_img4/image_10.png',
            92: 'assets/glasses_img4/image_11.png',
            100: 'assets/glasses_img4/image_11.png'
        },
        result6: {
            0: 'assets/panda/image_0.png',
            8: 'assets/panda/image_1.png',
            17: 'assets/panda/image_2.png',
            25: 'assets/panda/image_3.png',
            33: 'assets/panda/image_4.png',
            42: 'assets/panda/image_5.png',
            50: 'assets/panda/image_6.png',
            58: 'assets/panda/image_7.png',
            67: 'assets/panda/image_8.png',
            75: 'assets/panda/image_9.png',
            83: 'assets/panda/image_10.png',
            92: 'assets/panda/image_11.png',
            100: 'assets/panda/image_11.png'
        }
    };
    
    // Setup each result slider
    setupResultSlider('result1', 'result1-slider', 'result1-image', resultMappings.result1);
    setupResultSlider('result2', 'result2-slider', 'result2-image', resultMappings.result2);
    setupResultSlider('result3', 'result3-slider', 'result3-image', resultMappings.result3);
    setupResultSlider('result4', 'result4-slider', 'result4-image', resultMappings.result4);
    setupResultSlider('result5', 'result5-slider', 'result5-image', resultMappings.result5);
    setupResultSlider('result6', 'result6-slider', 'result6-image', resultMappings.result6);
});

function setupResultSlider(type, sliderId, imageId, imageMapping) {
    const slider = document.getElementById(sliderId);
    const image = document.getElementById(imageId);
    
    if (slider && image) {
        slider.addEventListener('input', function() {
            const sliderValue = parseInt(this.value);
            
            // Find the closest available image
            const availableValues = Object.keys(imageMapping).map(Number).sort((a, b) => a - b);
            let closestValue = availableValues[0];
            
            for (let i = 0; i < availableValues.length; i++) {
                if (Math.abs(availableValues[i] - sliderValue) < Math.abs(closestValue - sliderValue)) {
                    closestValue = availableValues[i];
                }
            }
            
            const imageSrc = imageMapping[closestValue];
            
            if (imageSrc) {
                // Add subtle fade effect during image change
                image.style.opacity = '0.8';
                
                // Change image source
                image.src = imageSrc;
                
                // Handle image load to restore opacity
                image.onload = function() {
                    this.style.opacity = '1';
                };
                
                // Handle image error (fallback to first image)
                image.onerror = function() {
                    this.src = imageMapping[0];
                    this.style.opacity = '1';
                };
            }
        });
        
        // Initialize with default image
        image.src = imageMapping[0];
    }
}

// Interactive Examples Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Define image mappings for the "Strength Controlled Image Editing" section
    const strengthControlledMappings = {
        'horse-uncle': {
            0: 'assets/horse_uncle/image_0.png',
            8: 'assets/horse_uncle/image_1.png',
            17: 'assets/horse_uncle/image_2.png',
            25: 'assets/horse_uncle/image_3.png',
            33: 'assets/horse_uncle/image_4.png',
            42: 'assets/horse_uncle/image_5.png',
            50: 'assets/horse_uncle/image_6.png',
            58: 'assets/horse_uncle/image_7.png',
            67: 'assets/horse_uncle/image_8.png',
            75: 'assets/horse_uncle/image_9.png',
            83: 'assets/horse_uncle/image_10.png',
            92: 'assets/horse_uncle/image_11.png',
            100: 'assets/horse_uncle/image_11.png'
        },
        'car-resize': {
            0: 'assets/car_reduce_size2/image_0.png',
            8: 'assets/car_reduce_size2/image_1.png',
            17: 'assets/car_reduce_size2/image_2.png',
            25: 'assets/car_reduce_size2/image_3.png',
            33: 'assets/car_reduce_size2/image_4.png',
            42: 'assets/car_reduce_size2/image_5.png',
            50: 'assets/car_reduce_size2/image_6.png',
            58: 'assets/car_reduce_size2/image_7.png',
            67: 'assets/car_reduce_size2/image_8.png',
            75: 'assets/car_reduce_size2/image_9.png',
            83: 'assets/car_reduce_size2/image_10.png',
            92: 'assets/car_reduce_size2/image_11.png',
            100: 'assets/car_reduce_size2/image_11.png'
        },
        'person-blur-red-hairs': {
            0: 'assets/person_blur_red_hairs/image_0.png',
            8: 'assets/person_blur_red_hairs/image_1.png',
            17: 'assets/person_blur_red_hairs/image_2.png',
            25: 'assets/person_blur_red_hairs/image_3.png',
            33: 'assets/person_blur_red_hairs/image_4.png',
            42: 'assets/person_blur_red_hairs/image_5.png',
            50: 'assets/person_blur_red_hairs/image_6.png',
            58: 'assets/person_blur_red_hairs/image_7.png',
            67: 'assets/person_blur_red_hairs/image_8.png',
            75: 'assets/person_blur_red_hairs/image_9.png',
            83: 'assets/person_blur_red_hairs/image_10.png',
            92: 'assets/person_blur_red_hairs/image_11.png',
            100: 'assets/person_blur_red_hairs/image_11.png'
        }
    };

    // Setup the "Strength Controlled Image Editing" sliders
    setupExampleSlider('horse-uncle', 'horse-uncle-slider', 'horse-uncle-image', strengthControlledMappings['horse-uncle']);
    setupExampleSlider('car-resize', 'car-resize-slider', 'car-resize-image', strengthControlledMappings['car-resize']);
    setupExampleSlider('person-blur-red-hairs', 'person-blur-red-hairs-slider', 'person-blur-red-hairs-image', strengthControlledMappings['person-blur-red-hairs']);

    // Define image mappings for all 15 results
    const exampleMappings = {
        'jacket-leather': {
            0: 'assets/jacket_leather/image_0.png',
            8: 'assets/jacket_leather/image_1.png',
            17: 'assets/jacket_leather/image_2.png',
            25: 'assets/jacket_leather/image_3.png',
            33: 'assets/jacket_leather/image_4.png',
            42: 'assets/jacket_leather/image_5.png',
            50: 'assets/jacket_leather/image_6.png',
            58: 'assets/jacket_leather/image_7.png',
            67: 'assets/jacket_leather/image_8.png',
            75: 'assets/jacket_leather/image_9.png',
            83: 'assets/jacket_leather/image_10.png',
            92: 'assets/jacket_leather/image_11.png',
            100: 'assets/jacket_leather/image_11.png'
        },
        'glasses-aviator': {
            0: 'assets/glasses_aviator/image_0.png',
            8: 'assets/glasses_aviator/image_1.png',
            17: 'assets/glasses_aviator/image_2.png',
            25: 'assets/glasses_aviator/image_3.png',
            33: 'assets/glasses_aviator/image_4.png',
            42: 'assets/glasses_aviator/image_5.png',
            50: 'assets/glasses_aviator/image_6.png',
            58: 'assets/glasses_aviator/image_7.png',
            67: 'assets/glasses_aviator/image_8.png',
            75: 'assets/glasses_aviator/image_9.png',
            83: 'assets/glasses_aviator/image_10.png',
            92: 'assets/glasses_aviator/image_11.png',
            100: 'assets/glasses_aviator/image_11.png'
        },
        'lamp-yellow': {
            0: 'assets/lamp_yellow/image_0.png',
            9: 'assets/lamp_yellow/image_1.png',
            18: 'assets/lamp_yellow/image_2.png',
            27: 'assets/lamp_yellow/image_3.png',
            36: 'assets/lamp_yellow/image_4.png',
            45: 'assets/lamp_yellow/image_5.png',
            55: 'assets/lamp_yellow/image_6.png',
            64: 'assets/lamp_yellow/image_7.png',
            73: 'assets/lamp_yellow/image_8.png',
            82: 'assets/lamp_yellow/image_9.png',
            91: 'assets/lamp_yellow/image_10.png',
            100: 'assets/lamp_yellow/image_10.png'
        },
        'man-fur-jacket-bike': {
            0: 'assets/man_fur_jacket_bike/image_0.png',
            12: 'assets/man_fur_jacket_bike/image_1.png',
            25: 'assets/man_fur_jacket_bike/image_2.png',
            37: 'assets/man_fur_jacket_bike/image_3.png',
            50: 'assets/man_fur_jacket_bike/image_4.png',
            62: 'assets/man_fur_jacket_bike/image_5.png',
            75: 'assets/man_fur_jacket_bike/image_6.png',
            87: 'assets/man_fur_jacket_bike/image_7.png',
            100: 'assets/man_fur_jacket_bike/image_8.png'
        },
        'model2-sunlight': {
            0: 'assets/model2_sunlight/image_0.png',
            8: 'assets/model2_sunlight/image_1.png',
            17: 'assets/model2_sunlight/image_2.png',
            25: 'assets/model2_sunlight/image_3.png',
            33: 'assets/model2_sunlight/image_4.png',
            42: 'assets/model2_sunlight/image_5.png',
            50: 'assets/model2_sunlight/image_6.png',
            58: 'assets/model2_sunlight/image_7.png',
            67: 'assets/model2_sunlight/image_8.png',
            75: 'assets/model2_sunlight/image_9.png',
            83: 'assets/model2_sunlight/image_10.png',
            92: 'assets/model2_sunlight/image_11.png',
            100: 'assets/model2_sunlight/image_11.png'
        },
        'panda-husky': {
            0: 'assets/panda_indoor2_husky_dog/image_0.png',
            8: 'assets/panda_indoor2_husky_dog/image_1.png',
            17: 'assets/panda_indoor2_husky_dog/image_2.png',
            25: 'assets/panda_indoor2_husky_dog/image_3.png',
            33: 'assets/panda_indoor2_husky_dog/image_4.png',
            42: 'assets/panda_indoor2_husky_dog/image_5.png',
            50: 'assets/panda_indoor2_husky_dog/image_6.png',
            58: 'assets/panda_indoor2_husky_dog/image_7.png',
            67: 'assets/panda_indoor2_husky_dog/image_8.png',
            75: 'assets/panda_indoor2_husky_dog/image_9.png',
            83: 'assets/panda_indoor2_husky_dog/image_10.png',
            92: 'assets/panda_indoor2_husky_dog/image_11.png',
            100: 'assets/panda_indoor2_husky_dog/image_11.png'
        },
        'person-blur-pixar': {
            0: 'assets/person_blur_pixar/image_0.png',
            8: 'assets/person_blur_pixar/image_1.png',
            17: 'assets/person_blur_pixar/image_2.png',
            25: 'assets/person_blur_pixar/image_3.png',
            33: 'assets/person_blur_pixar/image_4.png',
            42: 'assets/person_blur_pixar/image_5.png',
            50: 'assets/person_blur_pixar/image_6.png',
            58: 'assets/person_blur_pixar/image_7.png',
            67: 'assets/person_blur_pixar/image_8.png',
            75: 'assets/person_blur_pixar/image_9.png',
            83: 'assets/person_blur_pixar/image_10.png',
            92: 'assets/person_blur_pixar/image_11.png',
            100: 'assets/person_blur_pixar/image_11.png'
        },
        'tibbet-autumn': {
            0: 'assets/tibbet_autumn/image_0.png',
            8: 'assets/tibbet_autumn/image_1.png',
            17: 'assets/tibbet_autumn/image_2.png',
            25: 'assets/tibbet_autumn/image_3.png',
            33: 'assets/tibbet_autumn/image_4.png',
            42: 'assets/tibbet_autumn/image_5.png',
            50: 'assets/tibbet_autumn/image_6.png',
            58: 'assets/tibbet_autumn/image_7.png',
            67: 'assets/tibbet_autumn/image_8.png',
            75: 'assets/tibbet_autumn/image_9.png',
            83: 'assets/tibbet_autumn/image_10.png',
            92: 'assets/tibbet_autumn/image_11.png',
            100: 'assets/tibbet_autumn/image_11.png'
        },
        'venice-vegetation': {
            0: 'assets/venice1_Grow_vegetation_on_t_3/image_0.png',
            8: 'assets/venice1_Grow_vegetation_on_t_3/image_1.png',
            17: 'assets/venice1_Grow_vegetation_on_t_3/image_2.png',
            25: 'assets/venice1_Grow_vegetation_on_t_3/image_3.png',
            33: 'assets/venice1_Grow_vegetation_on_t_3/image_4.png',
            42: 'assets/venice1_Grow_vegetation_on_t_3/image_5.png',
            50: 'assets/venice1_Grow_vegetation_on_t_3/image_6.png',
            58: 'assets/venice1_Grow_vegetation_on_t_3/image_7.png',
            67: 'assets/venice1_Grow_vegetation_on_t_3/image_8.png',
            75: 'assets/venice1_Grow_vegetation_on_t_3/image_9.png',
            83: 'assets/venice1_Grow_vegetation_on_t_3/image_10.png',
            92: 'assets/venice1_Grow_vegetation_on_t_3/image_11.png',
            100: 'assets/venice1_Grow_vegetation_on_t_3/image_11.png'
        },
        'enfield-winter-snow': {
            0: 'assets/enfield3_winter_snow/image_0.png',
            8: 'assets/enfield3_winter_snow/image_1.png',
            17: 'assets/enfield3_winter_snow/image_2.png',
            25: 'assets/enfield3_winter_snow/image_3.png',
            33: 'assets/enfield3_winter_snow/image_4.png',
            42: 'assets/enfield3_winter_snow/image_5.png',
            50: 'assets/enfield3_winter_snow/image_6.png',
            58: 'assets/enfield3_winter_snow/image_7.png',
            67: 'assets/enfield3_winter_snow/image_8.png',
            75: 'assets/enfield3_winter_snow/image_9.png',
            83: 'assets/enfield3_winter_snow/image_10.png',
            92: 'assets/enfield3_winter_snow/image_11.png',
            100: 'assets/enfield3_winter_snow/image_11.png'
        }
    };
    
    // Setup all 15 sliders
    setupExampleSlider('jacket-leather', 'jacket-leather-slider', 'jacket-leather-image', exampleMappings['jacket-leather']);
    setupExampleSlider('glasses-aviator', 'glasses-aviator-slider', 'glasses-aviator-image', exampleMappings['glasses-aviator']);
    setupExampleSlider('lamp-yellow', 'lamp-yellow-slider', 'lamp-yellow-image', exampleMappings['lamp-yellow']);
    setupExampleSlider('man-fur-jacket-bike', 'man-fur-jacket-bike-slider', 'man-fur-jacket-bike-image', exampleMappings['man-fur-jacket-bike']);
    setupExampleSlider('model2-sunlight', 'model2-sunlight-slider', 'model2-sunlight-image', exampleMappings['model2-sunlight']);
    setupExampleSlider('panda-husky', 'panda-husky-slider', 'panda-husky-image', exampleMappings['panda-husky']);
    setupExampleSlider('person-blur-pixar', 'person-blur-pixar-slider', 'person-blur-pixar-image', exampleMappings['person-blur-pixar']);
    setupExampleSlider('tibbet-autumn', 'tibbet-autumn-slider', 'tibbet-autumn-image', exampleMappings['tibbet-autumn']);
    setupExampleSlider('venice-vegetation', 'venice-vegetation-slider', 'venice-vegetation-image', exampleMappings['venice-vegetation']);
    setupExampleSlider('enfield-winter-snow', 'enfield-winter-snow-slider', 'enfield-winter-snow-image', exampleMappings['enfield-winter-snow']);
});

// Copy BibTeX function
function copyBibTeX() {
    const bibtexText = document.querySelector('.bibtex-code').textContent;
    
    if (navigator.clipboard && window.isSecureContext) {
        // Use the modern clipboard API
        navigator.clipboard.writeText(bibtexText).then(() => {
            showCopyFeedback();
        }).catch(() => {
            fallbackCopyToClipboard(bibtexText);
        });
    } else {
        // Fallback for older browsers
        fallbackCopyToClipboard(bibtexText);
    }
}

function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showCopyFeedback();
    } catch (err) {
        console.error('Failed to copy text: ', err);
    }
    
    document.body.removeChild(textArea);
}

function showCopyFeedback() {
    const button = document.querySelector('.copy-bibtex-btn');
    const originalText = button.textContent;
    
    button.textContent = 'Copied!';
    button.style.background = '#27ae60';
    
    setTimeout(() => {
        button.textContent = originalText;
        button.style.background = '#3498db';
    }, 2000);
}

function setupExampleSlider(type, sliderId, imageId, imageMapping) {
    const slider = document.getElementById(sliderId);
    const image = document.getElementById(imageId);
    
    if (slider && image) {
        slider.addEventListener('input', function() {
            const sliderValue = parseInt(this.value);
            
            // Find the closest available image
            const availableValues = Object.keys(imageMapping).map(Number).sort((a, b) => a - b);
            let closestValue = availableValues[0];
            
            for (let i = 0; i < availableValues.length; i++) {
                if (Math.abs(availableValues[i] - sliderValue) < Math.abs(closestValue - sliderValue)) {
                    closestValue = availableValues[i];
                }
            }
            
            const imageSrc = imageMapping[closestValue];
            
            if (imageSrc) {
                // Add subtle fade effect during image change
                image.style.opacity = '0.8';
                
                // Change image source
                image.src = imageSrc;
                
                // Handle image load to restore opacity
                image.onload = function() {
                    this.style.opacity = '1';
                };
                
                // Handle image error (fallback to first image)
                image.onerror = function() {
                    this.src = imageMapping[0];
                    this.style.opacity = '1';
                };
            }
        });
        
        // Initialize with default image
        image.src = imageMapping[0];
    }
}
