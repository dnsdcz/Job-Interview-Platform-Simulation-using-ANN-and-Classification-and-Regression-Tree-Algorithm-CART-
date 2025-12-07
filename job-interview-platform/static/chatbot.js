console.log("chatbot.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  const chatContent = document.getElementById("chat-content");
  const responseBox = document.getElementById("answer-input");
  const submitBtn = document.getElementById("submit-btn");

  // old resume flow (not used in new UI but kept safe)
  const uploadStatus = document.getElementById("upload-status");
  const uploadBtn = document.getElementById("uploadResumeBtn");

  // hidden candidate info
  const candidateName = document.getElementById("candidate-name");
  const candidatePosition = document.getElementById("candidate-position");
  const candidateExperience = document.getElementById("candidate-experience");

  // result modal elements
  const resultModal = document.getElementById("resultModal");
  const resultTitle = document.getElementById("resultTitle");
  const resultStatusText = document.getElementById("resultStatusText");
  const resultPercentageText = document.getElementById("resultPercentageText");
  const resultSuggestionText = document.getElementById("resultSuggestionText");
  const resultViewSummaryBtn = document.getElementById("resultViewSummary");
  const resultDoneBtn = document.getElementById("resultDoneBtn");

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

  // init user data from hidden fields
  if (candidateName) userName = candidateName.value || "";
  if (candidatePosition) userPosition = candidatePosition.value || "";
  if (candidateExperience) userExperience = candidateExperience.value || "";

  function scrollToBottom() {
    if (!chatContent) return;
    chatContent.scrollTop = chatContent.scrollHeight;
  }

  function addBotMessage(html) {
    if (!chatContent) return;
    const msg = document.createElement("div");
    msg.classList.add("message", "bot");
    msg.innerHTML = html;
    chatContent.appendChild(msg);
    scrollToBottom();
  }

  function appendUserMessage(text) {
    if (!chatContent) return;
    const msg = document.createElement("div");
    msg.classList.add("message", "user");
    msg.textContent = text;
    chatContent.appendChild(msg);
    scrollToBottom();
  }

  // =========================
  // 1) LOAD QUESTIONS
  // =========================
  async function loadQuestions() {
    try {
      const res = await fetch("/get_questions");
      if (!res.ok) throw new Error("Failed to fetch questions");
      const data = await res.json();
      questions = data.questions || [];
      currentQuestionIndex = 0;
      scores = [];
      answeredQuestions = 0;
      finalScore = 0;
      hasInterviewEnded = false;

      if (responseBox) responseBox.disabled = false;
      if (submitBtn) submitBtn.disabled = false;

      if (questions.length === 0) {
        addBotMessage("No interview questions available at the moment.");
        finishInterview();
      } else {
        showNextQuestion();
      }
    } catch (err) {
      console.error("Questions fetch error:", err);
      if (uploadStatus) {
        uploadStatus.innerHTML =
          '<span class="error-message">Questions could not be loaded.</span>';
      } else {
        addBotMessage(
          '<span class="error-message">Questions could not be loaded.</span>'
        );
      }
    }
  }

  // called from HTML Start button
  window.startInterview = function () {
    loadQuestions();
  };

  // =========================
  // (optional) resume flow
  // =========================
  if (uploadBtn) {
    uploadBtn.addEventListener("click", async () => {
      const fileInput = document.getElementById("resume-file");
      const file = fileInput ? fileInput.files[0] : null;
      if (uploadStatus) uploadStatus.innerHTML = "";

      if (!file) {
        if (uploadStatus) {
          uploadStatus.innerHTML =
            '<span class="error-message">Please select a file to upload.</span>';
        }
        return;
      }

      if (!file.name.match(/\.(pdf|docx?)$/i)) {
        if (uploadStatus) {
          uploadStatus.innerHTML =
            '<span class="error-message">Only PDF or Word documents are allowed.</span>';
        }
        return;
      }

      const formData = new FormData();
      formData.append("resume", file);

      try {
        const res = await fetch("/upload_resume", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Upload failed");

        if (uploadStatus) {
          uploadStatus.innerHTML =
            '<span class="success-message">Resume uploaded successfully!</span>';
        }

        const info = document.querySelector(".user-info");
        if (info) {
          info.innerHTML = `
            <div class="info-row"><span class="info-label">Name:</span> ${data.name}</div>
            <div class="info-row"><span class="info-label">Experience:</span> ${data.experience} years</div>
            <div class="info-row"><span class="info-label">Position:</span> ${data.position}</div>
          `;
        }

        userName = data.name;
        userExperience = data.experience;
        userPosition = data.position;
        userSkills = data.skills || [];

        loadQuestions();
      } catch (err) {
        console.error("Upload error:", err);
        if (uploadStatus) {
          uploadStatus.innerHTML = `<span class="error-message">${err.message}</span>`;
        }
      }
    });
  }

  // =========================
  // suggestion helpers
  // =========================
  function shouldShowPdfSuggestion(position) {
    const allowedRoles = ["Business Analyst", "Project Manager", "Java Developer"];
    return allowedRoles.includes(position);
  }

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

      const btn = document.getElementById("showSuggestionBtn");
      if (btn) {
        btn.addEventListener("click", () => {
          const suggestionText =
            "This is a sample suggested answer for your study.";
          const suggestionDiv = document.getElementById("suggestionText");
          if (suggestionDiv) {
            suggestionDiv.textContent = suggestionText;
            suggestionDiv.style.display = "block";
          }
        });
      }
    }
  }

  // =========================
  // show next question
  // =========================
  function showNextQuestion() {
    if (currentQuestionIndex < questions.length) {
      const next = questions[currentQuestionIndex];
      addBotMessage(`<strong>Q${currentQuestionIndex + 1}:</strong> ${next}`);
      currentQuestionIndex++;
    } else {
      finishInterview();
    }
  }

  // =========================
  // submit answer
  // =========================
  async function submitAnswer() {
    if (hasInterviewEnded) {
      alert("The interview is already complete.");
      return;
    }

    if (!responseBox) return;

    const answer = responseBox.value.trim();
    if (!answer) {
      alert("Please type your answer!");
      return;
    }

    const currentQuestion = questions[currentQuestionIndex - 1];
    appendUserMessage(answer);

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Submitting...";
    }

    try {
      const res = await fetch("/score_answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, answer }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Scoring failed");

      const status = data.qualification_status || "Pending";
      const feedback = data.feedback || "No feedback provided.";

      scores.push({ question: currentQuestion, answer, qualificationStatus: status });
      questionAnswerPairs.push({ question: currentQuestion, answer });

      if (status === "Qualified") finalScore++;

      addBotMessageWithSuggestion(
        `<strong>Status:</strong> ${status}<br><em>${feedback}</em><hr>`,
        status
      );

      responseBox.value = "";
      answeredQuestions++;

      showNextQuestion();
    } catch (err) {
      console.error("Error scoring answer:", err);
      addBotMessage(
        '<div class="error-message">An error occurred while scoring the answer. Please try again.</div><hr>'
      );
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Send";
      }
    }
  }

  // exposed for onclick in HTML
  window.submitAnswer = submitAnswer;

  if (responseBox) {
    responseBox.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        submitAnswer();
      }
    });
  }

  // =========================
  // finish interview + modal
  // =========================
  function getAlternativeRoleSuggestion(position) {
    const p = (position || "").toLowerCase();
    if (p.includes("project")) {
      return "You might also explore roles like Business Analyst or Project Coordinator, where you can grow into full project management.";
    }
    if (p.includes("business") && p.includes("analyst")) {
      return "You might also fit roles such as Junior Project Manager or Operations Analyst.";
    }
    if (p.includes("java") || p.includes("developer")) {
      return "You might also consider roles like Junior Software Engineer or QA Automation Engineer.";
    }
    return "You may want to explore junior or training roles where you can build more experience while being mentored.";
  }

  function finishInterview() {
    addBotMessage("Interview Complete!");
    if (responseBox) responseBox.disabled = true;
    if (submitBtn) submitBtn.disabled = true;
    hasInterviewEnded = true;

    const qualifiedCount = scores.filter(
      (s) => s.qualificationStatus === "Qualified"
    ).length;
    const passThreshold = Math.ceil(scores.length * 0.7);
    finalResult = qualifiedCount >= passThreshold ? "Qualified" : "Not Qualified";
    const percent =
      scores.length > 0
        ? Math.round((qualifiedCount / scores.length) * 100)
        : 0;
    confidence = scores.length > 0 ? `${percent}%` : "N/A";

    const finalDiv = document.getElementById("final-result");
    if (finalDiv) {
      finalDiv.textContent = `Final Status: ${finalResult}`;
      finalDiv.style.display = "block";
    }
    const saveContainer = document.getElementById("save-container");
    const summaryLink = document.getElementById("summary-link");
    if (saveContainer) saveContainer.style.display = "block";
    if (summaryLink) summaryLink.style.display = "block";

    // show result modal
    if (resultModal) {
      if (resultTitle) {
        resultTitle.innerHTML =
          finalResult === "Qualified"
            ? '<i class="fa-regular fa-circle-check"></i> Congrats, you\'re qualified!'
            : '<i class="fa-regular fa-circle-xmark"></i> Interview result';
      }
      if (resultStatusText) {
        resultStatusText.innerHTML = `<strong>Status:</strong> ${finalResult}`;
      }
      if (resultPercentageText) {
        resultPercentageText.innerHTML = `<strong>Match score:</strong> ${percent}% (${qualifiedCount} of ${scores.length} answers qualified)`;
      }
      if (resultSuggestionText) {
        if (finalResult === "Qualified") {
          resultSuggestionText.textContent =
            "You’re a strong match for this role. You may proceed to the next step of the application.";
        } else {
          resultSuggestionText.textContent = getAlternativeRoleSuggestion(
            userPosition
          );
        }
      }
      resultModal.classList.add("show");
    }

    // send summary to backend for DB / HR page
    sendSummaryReport();
  }

  // =========================
  // save summary to /save_summary_report
  // =========================
  function sendSummaryReport() {
    const numericConfidence =
      typeof confidence === "string" && confidence.includes("%")
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
        confidence: parseFloat(numericConfidence),
        average_score:
          scores.length > 0 ? (finalScore / scores.length).toFixed(2) : 0,
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
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || "Failed to save summary report");
        }
        return res.json();
      })
      .then((data) => {
        console.log("Summary saved:", data.message);
        const saveBtn = document.getElementById("save");
        if (saveBtn) saveBtn.disabled = true;
        const uploadSection = document.querySelector(".upload-section");
        if (uploadSection) uploadSection.style.display = "none";
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      })
      .catch((err) => {
        console.error("Error saving summary:", err);
        addBotMessage(
          '<div class="error-message">Failed to save interview summary.</div>'
        );
      });
  }

  // view summary from bottom button
  const viewSummaryBtn = document.getElementById("view-summary");
  if (viewSummaryBtn) {
    viewSummaryBtn.addEventListener("click", function () {
      fetch("/summary_report", {
        method: "GET",
        credentials: "include",
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
  }

  // result modal buttons (no X, no backdrop close)
  if (resultViewSummaryBtn) {
    resultViewSummaryBtn.addEventListener("click", () => {
      window.location.href = "/summary_report";
    });
  }

  function closeResultModal() {
    if (resultModal) resultModal.classList.remove("show");
  }

  if (resultDoneBtn) {
    resultDoneBtn.addEventListener("click", () => {
      closeResultModal();
      const url = resultDoneBtn.dataset.redirectUrl || "/dashboard";
      window.location.href = url;
    });
  }

  // note: no click-outside close, and no X button wired
});
