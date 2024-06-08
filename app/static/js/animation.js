document.addEventListener("DOMContentLoaded", function() {
    const descriptionContainer = document.querySelector('.description-container');

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                descriptionContainer.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    });

    observer.observe(descriptionContainer);
});

document.addEventListener("DOMContentLoaded", function() {
    const descriptionContainer = document.querySelector('.app-description');

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    });

    observer.observe(descriptionContainer);
});
