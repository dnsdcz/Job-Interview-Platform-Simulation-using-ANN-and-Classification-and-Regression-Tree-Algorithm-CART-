console.log("chatbot.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  const chatContent = document.getElementById("chat-content");
  const responseBox = document.getElementById("answer-input");
  const uploadStatus = document.getElementById("upload-status");
  const submitBtn = document.getElementById("submit-btn");
  const uploadBtn = document.getElementById("uploadResumeBtn");

  let questions = [];
  let currentQuestionIndex = 0;
  let scores = [];
  let answeredQuestions = 0;
  let hasInterviewEnded = false;

  let userName = "";
  let userExperience = "";
  let userPosition = "";
  let userSkills = [];
  let questionAnswerPairs = [];
  let finalScore = 0;
  let finalResult = "";
  let confidence = "";

  function scrollToBottom() {
    chatContent.scrollTop = chatContent.scrollHeight;
  }

  function addBotMessage(html) {
    const msg = document.createElement("div");
    msg.classList.add("message", "bot");
    msg.innerHTML = html;
    chatContent.appendChild(msg);
    scrollToBottom();
  }

  function appendUserMessage(text) {
    const msg = document.createElement("div");
    msg.classList.add("message", "user");
    msg.textContent = text;
    chatContent.appendChild(msg);
    scrollToBottom();
  }

  async function loadQuestions() {
    try {
      const res = await fetch("/fetch_questions_after_resume");
      if (!res.ok) throw new Error("Failed to fetch questions");
      const data = await res.json();
      questions = data.questions || [];
      currentQuestionIndex = 0;
      scores = [];
      answeredQuestions = 0;
      finalScore = 0;
      hasInterviewEnded = false;
      responseBox.disabled = false;
      submitBtn.disabled = false;

      if (questions.length === 0) {
        addBotMessage("No interview questions available at the moment.");
        finishInterview();
      } else {
        showNextQuestion();
      }
    } catch (err) {
      console.error("Questions fetch error:", err);
      uploadStatus.innerHTML = '<span class="error-message">Questions could not be loaded.</span>';
    }
  }

  uploadBtn.addEventListener("click", async () => {
    const file = document.getElementById("resume-file").files[0];
    uploadStatus.innerHTML = "";

    if (!file) {
      uploadStatus.innerHTML = '<span class="error-message">Please select a file to upload.</span>';
      return;
    }

    if (!file.name.match(/\.(pdf|docx?)$/i)) {
      uploadStatus.innerHTML = '<span class="error-message">Only PDF or Word documents are allowed.</span>';
      return;
    }

    const formData = new FormData();
    formData.append("resume", file);

    try {
      const res = await fetch("/upload_resume", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");

      uploadStatus.innerHTML = '<span class="success-message">Resume uploaded successfully!</span>';
      document.querySelector(".user-info").innerHTML = `
        <div class="info-row"><span class="info-label">Name:</span> ${data.name}</div>
        <div class="info-row"><span class="info-label">Experience:</span> ${data.experience} years</div>
        <div class="info-row"><span class="info-label">Position:</span> ${data.position}</div>
      `;

      userName = data.name;
      userExperience = data.experience;
      userPosition = data.position;
      userSkills = data.skills || [];

      loadQuestions();
    } catch (err) {
      console.error("Upload error:", err);
      uploadStatus.innerHTML = `<span class="error-message">${err.message}</span>`;
    }
  });

  async function submitAnswer() {
    if (hasInterviewEnded) {
      alert("The interview is already complete.");
      return;
    }

    const answer = responseBox.value.trim();
    if (!answer) {
      alert("Please type your answer!");
      return;
    }

    const currentQuestion = questions[currentQuestionIndex - 1];
    appendUserMessage(answer);
    submitBtn.disabled = true;
    submitBtn.textContent = "Submitting...";

    try {
      const res = await fetch("/score_answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, answer })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Scoring failed");

      const status = data.qualification_status || "Pending";
      const feedback = data.feedback || "No feedback provided.";

      scores.push({ question: currentQuestion, answer, qualificationStatus: status });
      questionAnswerPairs.push({ question: currentQuestion, answer });

      if (status === "Qualified") finalScore++;

      addBotMessage(`<strong>Status:</strong> ${status}`);
      addBotMessage(`<em>${feedback}</em><hr>`);

      responseBox.value = "";
      answeredQuestions++;

      showNextQuestion();
    } catch (err) {
      console.error("Error scoring answer:", err);
      addBotMessage('<div class="error-message">An error occurred while scoring the answer. Please try again.</div><hr>');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit";
    }
  }

  submitBtn.addEventListener("click", submitAnswer);

  function showNextQuestion() {
    if (currentQuestionIndex < questions.length) {
      const next = questions[currentQuestionIndex];
      addBotMessage(`<strong>Q${currentQuestionIndex + 1}:</strong> ${next}`);
      currentQuestionIndex++;
    } else {
      finishInterview();
    }
  }

  function finishInterview() {
    addBotMessage("Interview Complete!");
    responseBox.disabled = true;
    submitBtn.disabled = true;
    hasInterviewEnded = true;

    const qualifiedCount = scores.filter((s) => s.qualificationStatus === "Qualified").length;
    const passThreshold = Math.ceil(scores.length * 0.7);
    finalResult = qualifiedCount >= passThreshold ? "Qualified" : "Not Qualified";
    confidence = scores.length > 0 ? `${Math.round((qualifiedCount / scores.length) * 100)}%` : "N/A";

    document.getElementById("final-result").textContent = `Final Status: ${finalResult}`;
    document.getElementById("final-result").style.display = "block";
    document.getElementById("save-container").style.display = "block";
    document.getElementById("summary-link").style.display = "block";

    const saveBtn = document.getElementById("save");
    if (saveBtn && !saveBtn.dataset.bound) {
      saveBtn.addEventListener("click", () => {
        alert("Interview results saved (already done automatically).");
      });
      saveBtn.dataset.bound = "true";
    }

    sendSummaryReport();
  }

  function sendSummaryReport() {
    const numericConfidence = typeof confidence === "string" && confidence.includes("%")
      ? parseFloat(confidence.replace("%", ""))
      : confidence;

    fetch("/save_summary_report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_name: userName,
        position: userPosition,
        experience: userExperience,
        skills: userSkills,
        qualification_status: finalResult,
        confidence: parseFloat(numericConfidence),  // 🔧 Ensure it's a number
        average_score: scores.length > 0 ? (finalScore / scores.length).toFixed(2) : 0,
        assessment_data: scores,
        advice: scores.map((s) => ({
          question: s.question,
          suggestion:
            s.qualificationStatus === "Qualified"
              ? "Well answered, keep it up."
              : "Review this topic to improve your knowledge.",
        })),
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.error || "Failed to save summary report");
        }
        return res.json();
      })
      .then((data) => {
        console.log("Summary saved:", data.message);
        document.getElementById("save").disabled = true;
        const viewSummaryBtn = document.getElementById("view-summary-btn");
        if (viewSummaryBtn) viewSummaryBtn.disabled = false;
        const uploadSection = document.querySelector(".upload-section");
        if (uploadSection) uploadSection.style.display = "none";
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      })
      .catch((err) => {
        console.error("Error saving summary:", err);
        addBotMessage('<div class="error-message">Failed to save interview summary.</div>');
      });
  }

  document.getElementById("view-summary")?.addEventListener("click", function () {
    fetch("/summary_report", {
      method: "GET",
      credentials: "include"
    })
    .then((response) => {
      if (!response.ok) throw new Error("Failed to load summary");
      window.location.href = "/summary_report";
    })
    .catch((error) => {
      console.error("Failed to fetch summary:", error);
      alert("Could not load summary report at this time.");
    });
  });

  // Show PDF suggestion only for selected positions
  function shouldShowPdfSuggestion(position) {
    const allowedRoles = ["Business Analyst", "Project Manager", "Java Developer"];
    return allowedRoles.includes(position);
  }

  // When displaying bot message after scoring, check if suggestion needed
  function addBotMessageWithSuggestion(html, status) {
    addBotMessage(html);

    if (status === "Not Qualified" && shouldShowPdfSuggestion(userPosition)) {
      addBotMessage(
        `<div style="margin-top:10px; font-style: italic; color:#777;">
          Your answer seems off. Here's a suggested answer you can study:<br>
          <button id="showSuggestionBtn" style="margin-top:5px;">Show Suggested Answer</button>
          <div id="suggestionText" style="display:none; margin-top:5px; padding:10px; background:#f0f0f0; border-radius:5px;"></div>
        </div>`
      );

      document.getElementById("showSuggestionBtn").addEventListener("click", () => {
        // Here you would fetch or show the stored suggestion for that question
        const suggestionText = "This is a sample suggested answer for your study.";
        const suggestionDiv = document.getElementById("suggestionText");
        suggestionDiv.textContent = suggestionText;
        suggestionDiv.style.display = "block";
      });
    }
  }

  // Override submitAnswer to use addBotMessageWithSuggestion instead of addBotMessage for feedback
  async function submitAnswer() {
    if (hasInterviewEnded) {
      alert("The interview is already complete.");
      return;
    }

    const answer = responseBox.value.trim();
    if (!answer) {
      alert("Please type your answer!");
      return;
    }

    const currentQuestion = questions[currentQuestionIndex - 1];
    appendUserMessage(answer);
    submitBtn.disabled = true;
    submitBtn.textContent = "Submitting...";

    try {
      const res = await fetch("/score_answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, answer })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Scoring failed");

      const status = data.qualification_status || "Pending";
      const feedback = data.feedback || "No feedback provided.";

      scores.push({ question: currentQuestion, answer, qualificationStatus: status });
      questionAnswerPairs.push({ question: currentQuestion, answer });

      if (status === "Qualified") finalScore++;

      addBotMessageWithSuggestion(`<strong>Status:</strong> ${status}<br><em>${feedback}</em><hr>`, status);

      responseBox.value = "";
      answeredQuestions++;

      showNextQuestion();
    } catch (err) {
      console.error("Error scoring answer:", err);
      addBotMessage('<div class="error-message">An error occurred while scoring the answer. Please try again.</div><hr>');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit";
    }
  }

});
