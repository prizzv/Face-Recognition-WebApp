class ExpressError extends Error {
    constructor(message, statusCode) {
        super();
        this.message = message;
        this.statusCode = statusCode;
        console.log(message + ". code: " + statusCode);
    }
}

module.exports = ExpressError;