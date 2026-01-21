document.getElementById('send-btn').addEventListener('click', function() {
    let userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    // Display user input in chat
    let chatOutput = document.getElementById('chat-output');
    chatOutput.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;

    // Send request to backend
    fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: userInput})
    })
    .then(response => response.json())
    .then(data => {
        chatOutput.innerHTML += `<p><strong>Bot:</strong> ${data.answer}</p>`;
    });

    document.getElementById('user-input').value = '';
});

document.getElementById('connect-email').addEventListener('click', function() {
    window.location.href = "mailto:college@domain.com";
});
