const messagesEl = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const input = document.querySelector("#input");
const sendButton = document.querySelector("#send");
const statusEl = document.querySelector("#status");
const promptButtons = document.querySelectorAll("[data-prompt]");
const qaDialog = document.querySelector("#qa-dialog");
const qaTitle = document.querySelector("#qa-title");
const qaQuestion = document.querySelector("#qa-question");
const qaAnswer = document.querySelector("#qa-answer");
const qaLink = document.querySelector("#qa-link");
const qaType = document.querySelector("#qa-type");
const API_URL = "/chat";

const conversation = [];
let latestUserQuestion = "";

function setStatus(text) {
  statusEl.textContent = text;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createMessage(role, text, recommendations = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "S";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  bubble.appendChild(paragraph);

  if (recommendations.length > 0) {
    bubble.appendChild(createRecommendations(recommendations));
  }

  article.appendChild(avatar);
  article.appendChild(bubble);
  messagesEl.appendChild(article);
  scrollToBottom();
}

function createRecommendations(recommendations) {
  const list = document.createElement("div");
  list.className = "recommendations";

  recommendations.forEach((item) => {
    const card = document.createElement("div");
    card.className = "recommendation";

    const link = document.createElement("a");
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = item.name;

    const type = document.createElement("span");
    type.className = "test-type";
    type.textContent = item.test_type || "?";

    const details = document.createElement("button");
    details.className = "details-button";
    details.type = "button";
    details.textContent = "Why";
    details.addEventListener("click", () => {
      openAssessmentNote(item);
    });

    card.appendChild(link);
    card.appendChild(type);
    card.appendChild(details);
    list.appendChild(card);
  });

  return list;
}

function openAssessmentNote(item) {
  qaTitle.textContent = item.name;
  qaQuestion.textContent = latestUserQuestion || "Why is this assessment in the shortlist?";
  qaAnswer.textContent = buildAssessmentNote(item);
  qaLink.href = item.url;
  qaType.textContent = item.test_type || "";
  qaDialog.showModal();
}

function buildAssessmentNote(item) {
  const typeText = {
    A: "ability and aptitude",
    B: "biodata or situational judgment",
    C: "competency",
    D: "development and 360",
    E: "assessment exercise",
    K: "knowledge and skills",
    P: "personality and behavior",
    S: "simulation",
  };

  const labels = String(item.test_type || "")
    .split(",")
    .map((code) => typeText[code.trim()] || code.trim())
    .filter(Boolean);

  const labelText = labels.length > 0 ? labels.join(" + ") : "catalog";

  return `${item.name} was included as a ${labelText} assessment from the SHL catalog. Its URL and name come from the retrieved catalog metadata, and it was selected by the recommendation pipeline for the current conversation context.`;
}

async function sendMessage(text) {
  const content = text.trim();

  if (!content) {
    return;
  }

  conversation.push({
    role: "user",
    content,
  });
  latestUserQuestion = content;

  createMessage("user", content);
  input.value = "";
  input.focus();
  sendButton.disabled = true;
  setStatus("Thinking");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: conversation,
      }),
    });

    if (!response.ok) {
      let detail = `Request failed with ${response.status}`;

      try {
        const errorData = await response.json();

        if (errorData.detail) {
          detail = errorData.detail;
        }
      } catch (_) {
        // Keep the status-only message when the backend sends non-JSON.
      }

      throw new Error(detail);
    }

    const data = await response.json();
    const assistantText = data.reply || "I could not produce a reply.";

    conversation.push({
      role: "assistant",
      content: assistantText,
    });

    createMessage(
      "assistant",
      assistantText,
      data.recommendations || []
    );
    setStatus(data.end_of_conversation ? "Complete" : "Ready");
  } catch (error) {
    createMessage(
      "assistant",
      `The backend returned an error: ${error.message}`
    );
    setStatus("Error");
  } finally {
    sendButton.disabled = false;
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(input.value);
  }
});

promptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
});
