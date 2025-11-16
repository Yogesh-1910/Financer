// finance-manager-backend/routes/authRoutes.js
const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { User } = require('../models'); // Import the User model

// --- Route for User Signup ---
// POST /api/auth/signup
router.post('/signup', async (req, res) => {
  const { fullName, username, password } = req.body;

  // Basic validation
  if (!fullName || !username || !password) {
    return res.status(400).json({ msg: 'Please enter all required fields: full name, username, and password.' });
  }

  try {
    // Check if user already exists
    let user = await User.findOne({ where: { username } });
    if (user) {
      return res.status(400).json({ msg: 'A user with that username already exists.' });
    }

    // Create new user instance (password will be hashed by the model hook)
    user = await User.create({
      fullName,
      username,
      password,
    });

    // Create JWT Payload
    const payload = {
      user: {
        id: user.id,
        username: user.username,
      },
    };

    // Sign the token
    jwt.sign(
      payload,
      process.env.JWT_SECRET,
      { expiresIn: '5h' }, // Token expires in 5 hours
      (err, token) => {
        if (err) throw err;
        const { password, ...userWithoutPassword } = user.get({ plain: true });
        res.status(201).json({
          token,
          user: userWithoutPassword
        });
      }
    );
  } catch (err) {
    console.error('Signup Error:', err.message);
    res.status(500).send('Server error during signup');
  }
});

// --- Route for User Login ---
// POST /api/auth/login
router.post('/login', async (req, res) => {
  const { username, password } = req.body;

  // Basic validation
  if (!username || !password) {
    return res.status(400).json({ msg: 'Please provide both username and password.' });
  }

  try {
    // Check if user exists
    const user = await User.findOne({ where: { username } });
    if (!user) {
      return res.status(400).json({ msg: 'Invalid credentials. User not found.' });
    }

    // Compare entered password with stored hashed password
    // The isValidPassword method is defined in your User model
    const isMatch = await user.isValidPassword(password);
    if (!isMatch) {
      return res.status(400).json({ msg: 'Invalid credentials. Incorrect password.' });
    }

    // User is valid, create JWT payload
    const payload = {
      user: {
        id: user.id,
        username: user.username,
      },
    };

    // Sign the token
    jwt.sign(
      payload,
      process.env.JWT_SECRET,
      { expiresIn: '5h' },
      (err, token) => {
        if (err) throw err;
        const { password, ...userWithoutPassword } = user.get({ plain: true });
        res.json({
          token,
          user: userWithoutPassword
        });
      }
    );
  } catch (err) {
    console.error('Login Error:', err.message);
    res.status(500).send('Server error during login');
  }
});

// This is the most important part that was missing.
// It exports the router so `server.js` can use it.
module.exports = router;