function togglePasswordVisibility() {
    const passwordInput = document.getElementById('password');
    const togglePasswordIcon = document.getElementById('toggle-password');
    
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        togglePasswordIcon.textContent = '🙈';
    } else {
        passwordInput.type = 'password';
        togglePasswordIcon.textContent = '👁️';
    }
}
