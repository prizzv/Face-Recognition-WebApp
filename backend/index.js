const express = require("express");
const app = express();
const path = require("path")
const Webcam = require("node-webcam");
const fs = require("fs");

// Set public as static directory
app.use(express.static('public'));

app.set('views', path.join(__dirname, '/views'))

// Use ejs as template engine
app.set('view engine', 'ejs');
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

const webcamOptions = {
    width: 1280,
    height: 720,
    quality: 100,
    delay: 0,
    saveShots: false,
    output: 'jpeg',
    device: false,
    callbackReturn: 'buffer',
    verbose: false
};


// Create the webcam instance
const webcam = Webcam.create(webcamOptions);

// Take a snapshot from the webcam
webcam.capture("test_picture", function (err, data) {
    if (err) {
        console.error('Error capturing image:', err);

        return;
    }
    
    const imagePath = path.join(__dirname, 'public', 'images', 'image.jpg');
    console.log(imagePath)

    // Save the image data to a file on disk
    fs.writeFile(imagePath, data, function (err) {
        if (err) {
            console.error('Error saving image:', err);
            return;
        }
        console.log('Image saved successfully!');
    });
});

// Render main template
app.get('/', (req, res) => {

    res.render('main')
})

// Server setup
app.listen(3000, () => {
    console.log("The server started running on port 3000")
});
