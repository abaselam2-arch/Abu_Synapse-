
import "dotenv/config";

import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import OpenAI from "openai";
import path from "node:path";
import { fileURLToPath } from "node:url";

const app = express();

const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.OPENAI_MODEL || "gpt-5-mini";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --------------------------------------------------
// ENVIRONMENT VALIDATION
// --------------------------------------------------

if (!process.env.OPENAI_API_KEY) {
  console.error("Missing OPENAI_API_KEY environment variable.");
  process.exit(1);
}

if (!process.env.FRONTEND_ORIGIN) {
  console.error("Missing FRONTEND_ORIGIN environment variable.");
  process.exit(1);
}

// --------------------------------------------------
// OPENAI CLIENT
// --------------------------------------------------

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// --------------------------------------------------
// SECURITY CONFIGURATION
// --------------------------------------------------

app.disable("x-powered-by");

app.use(
  helmet({
    contentSecurityPolicy: false
  })
);

app.use(
  cors({
    origin: process.env.FRONTEND_ORIGIN,
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"],
    credentials: false
  })
);

app.use(
  express.json({
    limit: "20kb"
  })
);

// --------------------------------------------------
// RATE LIMITING
// --------------------------------------------------

const aiLimiter = rateLimit({
  windowMs: 60 * 1000,
  limit: 10,
  standardHeaders: "draft-8",
  legacyHeaders: false,
  message: {
    error: "Too many requests. Please wait one minute."
  }
});

// --------------------------------------------------
// CONSTANTS
// --------------------------------------------------

const MAX_MESSAGE_LENGTH = 2000;
const MAX_HISTORY_ITEMS = 8;
const MAX_HISTORY_MESSAGE_LENGTH = 1200;

const ALLOWED_SUBJECTS = new Set([
  "Mathematics",
  "Physics",
  "Chemistry",
  "Biology",
  "General"
]);

const ALLOWED_LEVELS = new Set([
  "Beginner",
  "Intermediate",
  "Advanced"
]);

const ALLOWED_MODES = new Set([
  "Explain",
  "Hint",
  "Practice",
  "Review"
]);

// --------------------------------------------------
// TUTOR INSTRUCTIONS
// --------------------------------------------------

const TUTOR_INSTRUCTIONS = `
You are SYNAPSE AI Tutor, a patient and adaptive educational assistant.

Your main goal is to help students understand concepts instead of simply
giving them final answers.

TEACHING RULES:
1. Use clear, beginner-friendly language.
2. Adapt explanations to the student's education level.
3. Ask one useful follow-up question when appropriate.
4. Use the Socratic method: guide the learner with small steps.
5. For mathematics, show the reasoning and verify calculations.
6. If the student is confused, simplify the explanation.
7. Use examples related to the selected subject.
8. Never shame, insult, or discourage the student.
9. If the student asks for an answer, explain the method before or together
   with the answer, unless they explicitly request a short answer.
10. Do not claim that you have access to private student records.
11. Do not invent textbooks, exam policies, grades, or official facts.
12. If the question is outside education, politely redirect it.
13. Do not request passwords, API keys, identity documents, or sensitive data.
14. Do not provide dangerous instructions.
15. Do not pretend to be a human teacher or a licensed professional.

RESPONSE FORMAT:
- Start with a direct and helpful response.
- Use short sections when useful.
- Give a simple example when it improves understanding.
- Finish with a short check-for-understanding question when appropriate.
- Avoid unnecessary long responses.
`;

// --------------------------------------------------
// INPUT VALIDATION HELPERS
// --------------------------------------------------

function cleanText(value, maxLength) {
  if (typeof value !== "string") {
    return "";
  }

  return value
    .replace(/\u0000/g, "")
    .trim()
    .slice(0, maxLength);
}

function normalizeHistory(history) {
  if (!Array.isArray(history)) {
    return [];
  }

  return history
    .slice(-MAX_HISTORY_ITEMS)
    .filter((item) => item && typeof item === "object")
    .map((item) => {
      const role = item.role === "assistant" ? "assistant" : "user";

      return {
        role,
        content: cleanText(
          item.content,
          MAX_HISTORY_MESSAGE_LENGTH
        )
      };
    })
    .filter((item) => item.content.length > 0);
}

function isAllowedValue(value, allowedValues) {
  return allowedValues.has(value);
}

// --------------------------------------------------
// HEALTH CHECK
// --------------------------------------------------

app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    service: "SYNAPSE AI Tutor",
    time: new Date().toISOString()
  });
});

// --------------------------------------------------
// AI TUTOR ENDPOINT
// --------------------------------------------------

app.post("/api/tutor", aiLimiter, async (req, res) => {
  try {
    const body = req.body || {};

    const message = cleanText(
      body.message,
      MAX_MESSAGE_LENGTH
    );

    const subject = cleanText(body.subject, 50);
    const level = cleanText(body.level, 50);
    const mode = cleanText(body.mode, 50);
    const concept = cleanText(body.concept, 100);
    const history = normalizeHistory(body.history);

    if (!message) {
      return res.status(400).json({
        error: "Please enter a question."
      });
    }

    if (!isAllowedValue(subject, ALLOWED_SUBJECTS)) {
      return res.status(400).json({
        error: "Invalid subject."
      });
    }

    if (!isAllowedValue(level, ALLOWED_LEVELS)) {
      return res.status(400).json({
        error: "Invalid learning level."
      });
    }

    if (!isAllowedValue(mode, ALLOWED_MODES)) {
      return res.status(400).json({
        error: "Invalid tutor mode."
      });
    }

    const context = [
      `Subject: ${subject}`,
      `Learning level: ${level}`,
      `Tutor mode: ${mode}`,
      `Current concept: ${concept || "Not specified"}`,
      "",
      "Student message:",
      message
    ].join("\n");

    const input = [
      ...history,
      {
        role: "user",
        content: context
      }
    ];

    const response = await openai.responses.create({
      model: MODEL,
      instructions: TUTOR_INSTRUCTIONS,
      input,
      store: false,
      max_output_tokens: 700
    });

    const answer = cleanText(
      response.output_text || "",
      6000
    );

    if (!answer) {
      return res.status(502).json({
        error: "The AI returned an empty response."
      });
    }

    return res.json({
      success: true,
      answer,
      model: MODEL,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error("AI tutor request failed:", {
      name: error?.name,
      status: error?.status,
      message: error?.message
    });

    if (error?.status === 429) {
      return res.status(429).json({
        error: "The AI service is busy. Please try again later."
      });
    }

    return res.status(500).json({
      error: "The tutor is temporarily unavailable."
    });
  }
});

// --------------------------------------------------
// FRONTEND STATIC FILES
// --------------------------------------------------

app.use(express.static(path.join(__dirname, "public")));

// --------------------------------------------------
// UNKNOWN API ROUTES
// --------------------------------------------------

app.use("/api", (req, res) => {
  res.status(404).json({
    error: "API route not found."
  });
});

// --------------------------------------------------
// GENERAL ERROR HANDLER
// --------------------------------------------------

app.use((error, req, res, next) => {
  console.error("Server error:", error?.message);

  res.status(500).json({
    error: "Unexpected server error."
  });
});

// --------------------------------------------------
// START SERVER
// --------------------------------------------------

app.listen(PORT, "0.0.0.0", () => {
  console.log(`SYNAPSE backend running on port ${PORT}`);
});
