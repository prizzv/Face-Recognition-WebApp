// This file is used to get the user's image from the webcam 


// Get the login and register forms
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');

// Hide the login form by default
registerForm.style.display = 'none';

function showLoginForm() {
    loginForm.style.display = 'flex';
    registerForm.style.display = 'none';
}

function showRegisterForm() {
    loginForm.style.display = 'none';
    registerForm.style.display = 'flex';
}

// Call the appropriate function when the links are clicked
document.getElementById('register-link').addEventListener('click', showRegisterForm);
document.getElementById('login-link').addEventListener('click', showLoginForm);


// Get the video element 
var video = document.getElementById('videoElement');
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();

            // Take a picture when the user clicks a button
            document.getElementById('login-submit-button').addEventListener('click', async function () {
                // Draw the video frame onto the canvas
                var canvas = document.getElementById('canvas');
                var context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Convert the canvas image to a data URL
                var dataURL = canvas.toDataURL();

                try {
                    // await axios.post('/getUserImage', {
                    //     imageData: dataURL
                    // })
                    predict(dataURL);

                } catch (error) {
                    console.log(error);
                }
            });

        }).catch(function (error) {

            console.log("Something went wrong!")
        })
} else {
    console.log("No camera found")
}

// Setting up tfjs with the model we downloaded
tf.loadLayersModel('model/model.json')
    .then(function (model) {
        window.model = model;
    });

// Predict function
var predict = function (input) {
    if (window.model) {
        window.model.predict([tf.tensor(input)
            .reshape([1, 28, 28, 1])])
            .array().then(function (scores) {
                scores = scores[0];
                predicted = scores
                    .indexOf(Math.max(...scores));
                    console.log(predicted);});
    } else {
        // The model takes a bit to load, so if we are too fast, wait a bit and try again
        setTimeout(function () { predict(input) }, 50);
    }
}