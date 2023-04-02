const express = require("express");
const methodOverride = require('method-override')
const app = express();
const path = require("path")
const fs = require("fs");
const session = require('express-session')
const cookieParser = require('cookie-parser');

const axios = require('axios');

//To parse form data in POST request body:
app.use(express.urlencoded({ extended: true }))

// To parse incoming JSON in POST request body:
app.use(express.json())
app.set('views', path.join(__dirname, '/views'))
app.use(methodOverride('_method'))
app.set('views', path.join(__dirname, 'views'))
// Set public as static directory
app.use(express.static(path.join(__dirname, 'public')));

const wrapAsync = require('./utils/wrapAsync');
const ExpressError = require('./utils/ExpressError');

// Use ejs as template engine
app.set('view engine', 'ejs');

app.use(session({
    secret: 'notagoodsecret',
    resave: false,
    saveUninitialized: true
}))

// Render main template
app.get('/', (req, res) => {

    res.render('main')
})

app.get('/login', (req, res) => {

    res.render('login')
})
app.post('/getUserImage', wrapAsync(async(req, res,next) => {
    const { imageData } = req.body;
    
    function saveDataURLtoFile(dataURL, fileName) {
        // Remove the "data:image/png;base64," prefix from the dataURL
        const data = dataURL.replace(/^data:image\/\w+;base64,/, '');
        const buffer = Buffer.from(data, 'base64');

        fs.writeFile(fileName, buffer, function(error) {

          if (error) {
            console.log('Error saving image:', error);
          } else {
            console.log('Image saved successfully');
          }
        });
      }

      const fileName = 'public/images/image.png';
      
      saveDataURLtoFile(imageData, fileName);

    console.log("dataURL"+imageData)

    next(res.redirect('/'));
}))

app.post('/login', wrapAsync(async(req, res, next) => {
    const { username, password} = req.body;
    

    res.send('success222_page')
}))

app.get('/success_page', (req, res) => {

    res.render("success_page")
})

// Server setup
app.listen(3000, () => {
    console.log("The server started running on port 3000")
});
