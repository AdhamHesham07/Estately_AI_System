/* =============================================
   ESTATELY — Chatbot JavaScript
   Features: Mock responses, EN/AR i18n, typing
             indicator, auto-resize input, download
   ============================================= */

/* ─────────────────────────────────────────────
   i18n — Translations
   ───────────────────────────────────────────── */
const i18n = {
  en: {
    agentIntroTitle:   'Estately AI',
    agentIntroDesc:    "Your smart real estate assistant for Egypt's property market",
    onlineStatus:      'Online & Ready',
    newChatLabel:      'New Conversation',
    capabilitiesTitle: 'Key Capabilities',
    examplesTitle:     'Example Prompts',
    feat1Title: 'Smart Property Search',
    feat1Desc:  'Find listings using natural language across all of Egypt',
    feat2Title: 'Market Analytics',
    feat2Desc:  'Get real-time price trends and investment insights',
    feat3Title: 'AI Recommendations',
    feat3Desc:  'Personalized property matches based on your needs',
    feat4Title: 'Multilingual Support',
    feat4Desc:  'Ask in English or Arabic — we understand both',
    chip1: 'Find me an apartment in New Cairo under 5 million EGP',
    chip2: 'Compare villa prices in Sheikh Zayed vs Fifth Settlement',
    chip3: 'I need a 3-bedroom place near AUC for 8 million EGP',
    chip4: 'What are the best investment areas in Cairo right now?',
    welcomeTitle: 'Welcome to Estately AI',
    welcomeDesc:  'Ask me anything about properties in Egypt — search listings, compare prices, or get personalized recommendations.',
    chatAgentName:   'Estately AI Assistant',
    chatAgentStatus: 'Online — Real Estate Expert',
    inputPlaceholder: 'Describe your ideal property...',
    inputHint: 'Press Enter to send · Shift+Enter for new line',
    greetingMsg: "Hello! 👋 I'm **Estately AI**, your personal real estate assistant.\n\nI can help you:\n• 🔍 **Search** properties across Egypt\n• 📊 **Compare** prices & market trends\n• 🏆 **Recommend** the best matches for your budget\n• ✅ **Verify** official agents\n\nWhat are you looking for today?",
    typingText: 'Estately AI is thinking...',
    navHomeLink: 'Home',
    navPropertiesLink: 'Properties',
    navAgenciesLink: 'Agencies',
    navServicesLink: 'Services',
    navAboutLink: 'About',
    navContactLink: 'Contact',
    navBlogLink: 'Blog',
    navLoginLink: 'Login',
    footerAgency: 'Real Estate Agency',
    footerCompany: 'Company',
    footerProduct: 'Product',
    footerAreas: 'Areas',
    footerResources: 'Resources',
    ftOverview: 'Overview',
    ftPricing: 'Pricing',
    ftMarket: 'Marketplace',
    ftFeatures: 'Features',
    ftInteg: 'Integrations',
    ftCairo: 'New Cairo',
    ftZayed: 'El Sheikh Zayed',
    ftAlamein: 'Al Alamein',
    ftSokhna: 'Ain Sokhna',
    ftHekma: 'Ras El Hekma',
    ftMaadi: 'El Maadi',
    ftHelp: 'Help',
    ftSales: 'Sales',
    ftAds: 'Advertise',
    ftFaq: 'FAQ',
    footerCopyright: 'Copyright © 2025 Estately. All rights reserved.',
    ftTerms: 'Terms and Conditions',
    ftPrivacy: 'Privacy Policy',
    wc1Label: 'Find apartments in New Cairo',
    wc1Prompt: 'Find apartments in New Cairo',
    wc2Label: 'Villa prices in Maadi',
    wc2Prompt: 'Show me villa prices in Maadi',
    wc3Label: 'Best investment areas',
    wc3Prompt: 'Best areas to invest in Cairo',
    wc4Label: '3-bed under 6M EGP',
    wc4Prompt: '3-bedroom under 6 million EGP',
  },
  ar: {
    agentIntroTitle:   'إستيتلي AI',
    agentIntroDesc:    'مساعدك الذكي في سوق العقارات المصري',
    onlineStatus:      'متصل وجاهز',
    newChatLabel:      'محادثة جديدة',
    capabilitiesTitle: 'المميزات الرئيسية',
    examplesTitle:     'أمثلة على الأسئلة',
    feat1Title: 'بحث ذكي عن العقارات',
    feat1Desc:  'ابحث في قوائم العقارات بلغة طبيعية في جميع أنحاء مصر',
    feat2Title: 'تحليلات السوق',
    feat2Desc:  'احصل على اتجاهات الأسعار في الوقت الفعلي ورؤى الاستثمار',
    feat3Title: 'توصيات بالذكاء الاصطناعي',
    feat3Desc:  'عروض عقارية مخصصة بناءً على احتياجاتك',
    feat4Title: 'دعم متعدد اللغات',
    feat4Desc:  'اسأل بالعربية أو الإنجليزية — نفهم الاثنين',
    chip1: 'ابحث عن شقة في القاهرة الجديدة بأقل من 5 مليون',
    chip2: 'قارن أسعار الفيلات في الشيخ زايد والتجمع الخامس',
    chip3: 'أريد شقة 3 غرف قرب الجامعة الأمريكية بـ 8 مليون جنيه',
    chip4: 'ما هي أفضل مناطق الاستثمار في القاهرة الآن؟',
    welcomeTitle: 'أهلاً بك في إستيتلي AI',
    welcomeDesc:  'اسألني عن أي عقار في مصر — ابحث عن قوائم، قارن الأسعار، أو احصل على توصيات شخصية.',
    chatAgentName:   'مساعد إستيتلي AI',
    chatAgentStatus: 'متصل — خبير عقاري',
    inputPlaceholder: 'صف العقار المثالي الذي تبحث عنه...',
    inputHint: 'اضغط Enter للإرسال · Shift+Enter لسطر جديد',
    greetingMsg: "مرحباً! 👋 أنا **إستيتلي AI**، مساعدك الشخصي في مجال العقارات.\n\nأستطيع مساعدتك في:\n• 🔍 **البحث** عن عقارات في جميع أنحاء مصر\n• 📊 **مقارنة** الأسعار واتجاهات السوق\n• 🏆 **توصية** بأفضل الخيارات لميزانيتك\n• ✅ **التحقق** من الوكلاء الرسميين\n\nبماذا يمكنني مساعدتك اليوم؟",
    typingText: 'إستيتلي AI يفكر...',
    navHomeLink: 'الرئيسية',
    navPropertiesLink: 'العقارات',
    navAgenciesLink: 'الوكالات',
    navServicesLink: 'الخدمات',
    navAboutLink: 'من نحن',
    navContactLink: 'اتصل بنا',
    navBlogLink: 'المدونة',
    navLoginLink: 'تسجيل الدخول',
    footerAgency: 'وكالة عقارية',
    footerCompany: 'الشركة',
    footerProduct: 'المنتج',
    footerAreas: 'المناطق',
    footerResources: 'الموارد',
    ftOverview: 'نظرة عامة',
    ftPricing: 'الأسعار',
    ftMarket: 'السوق',
    ftFeatures: 'المميزات',
    ftInteg: 'التكامل',
    ftCairo: 'القاهرة الجديدة',
    ftZayed: 'الشيخ زايد',
    ftAlamein: 'العلمين',
    ftSokhna: 'العين السخنة',
    ftHekma: 'رأس الحكمة',
    ftMaadi: 'المعادي',
    ftHelp: 'المساعدة',
    ftSales: 'المبيعات',
    ftAds: 'أعلن معنا',
    ftFaq: 'الأسئلة الشائعة',
    footerCopyright: 'حقوق النشر © 2025 إستيتلي. جميع الحقوق محفوظة.',
    ftTerms: 'الشروط والأحكام',
    ftPrivacy: 'سياسة الخصوصية',
    wc1Label: 'شقق في القاهرة الجديدة',
    wc1Prompt: 'ابحث عن شقق في القاهرة الجديدة',
    wc2Label: 'أسعار الفيلات في المعادي',
    wc2Prompt: 'اعرض لي أسعار الفيلات في المعادي',
    wc3Label: 'أفضل مناطق الاستثمار',
    wc3Prompt: 'ما هي أفضل المناطق للاستثمار في القاهرة؟',
    wc4Label: '3 غرف بأقل من 6 مليون',
    wc4Prompt: 'شقة 3 غرف نوم بأقل من 6 مليون جنيه',
  }
};

/* ─────────────────────────────────────────────
   Mock responses (demo mode)
   ───────────────────────────────────────────── */
const mockResponses = {
  en: [
    "Great question! 🏠 Based on your criteria, I found **3 matching properties** in New Cairo:\n\n1. **Midtown Sky** — 5,500,000 EGP · 150 m² · 3 beds\n2. **Palm Garden** — 3,800,000 EGP · 180 m² · 3 beds\n3. **Elite Residence** — 4,200,000 EGP · 130 m² · 2 beds\n\nWould you like more details on any of these?",
    "📊 **Market Insight for New Cairo:**\n\nAverage price per m²: **26,500 EGP**\nYoY price change: **+12.3%**\n\nTop developers: SODIC, Palm Hills, Emaar Misr\n\nThe Fifth Settlement area shows the strongest growth. Would you like a deeper comparison?",
    "I can help you find the perfect property! Could you tell me more about:\n- **Budget range** (in EGP)\n- **Number of bedrooms** needed\n- **Preferred area** in Cairo\n- **Ready to move in** or off-plan?",
    "🏆 **Top AI Recommendation for you:**\n\nBased on similar searches, I highly recommend checking **Scenario Compound** in New Cairo — it offers great value, verified agents, and is near the AUC campus. Starting from **7,500,000 EGP** for a 3-bedroom unit.\n\nShall I connect you with a verified agent?",
    "✅ **Agent Verification:**\n\nI can help verify any agent's credentials. Please share the agent's **phone number** and I'll check their official registration status immediately.",
  ],
  ar: [
    "سؤال ممتاز! 🏠 بناءً على معاييرك، وجدت **3 عقارات مطابقة** في القاهرة الجديدة:\n\n1. **ميدتاون سكاي** — 5,500,000 جنيه · 150 م² · 3 غرف\n2. **بالم جاردن** — 3,800,000 جنيه · 180 م² · 3 غرف\n3. **إيليت ريزيدنس** — 4,200,000 جنيه · 130 م² · غرفتان\n\nهل تريد مزيداً من التفاصيل عن أي منها؟",
    "📊 **تحليل سوق القاهرة الجديدة:**\n\nمتوسط السعر لكل م²: **26,500 جنيه**\nالتغير السنوي: **+12.3%**\n\nأفضل المطورين: سوديك، بالم هيلز، إعمار مصر\n\nمنطقة التجمع الخامس تُظهر أقوى نمو. هل تريد مقارنة أعمق؟",
    "يسعدني مساعدتك في إيجاد العقار المثالي! هل يمكنك إخباري بـ:\n- **الميزانية** (بالجنيه المصري)\n- **عدد الغرف** المطلوبة\n- **المنطقة المفضلة** في القاهرة\n- **جاهز للسكن** أم على الخريطة؟",
  ]
};

/* ─────────────────────────────────────────────
   State
   ───────────────────────────────────────────── */
let currentLang    = 'en';
let messages       = [];
let isTyping       = false;
let messageIdCounter = 0;

/* ─────────────────────────────────────────────
   DOM refs
   ───────────────────────────────────────────── */
const messagesArea  = document.getElementById('messages-area');
const chatInput     = document.getElementById('chat-input');
const sendBtn       = document.getElementById('send-btn');
const welcomeState  = document.getElementById('welcome-state');
const charCount     = document.getElementById('char-count');

/* ─────────────────────────────────────────────
   Language switch
   ───────────────────────────────────────────── */
function setLang(lang) {
  currentLang = lang;
  const t = i18n[lang];
  const html = document.documentElement;

  // RTL / LTR
  html.dir  = lang === 'ar' ? 'rtl' : 'ltr';
  html.lang = lang;

  // Update all i18n-keyed elements
  const map = {
    'agent-intro-title':  t.agentIntroTitle,
    'agent-intro-desc':   t.agentIntroDesc,
    'online-status':      t.onlineStatus,
    'new-chat-label':     t.newChatLabel,
    'capabilities-title': t.capabilitiesTitle,
    'examples-title':     t.examplesTitle,
    'feat-1-title':       t.feat1Title,
    'feat-1-desc':        t.feat1Desc,
    'feat-2-title':       t.feat2Title,
    'feat-2-desc':        t.feat2Desc,
    'feat-3-title':       t.feat3Title,
    'feat-3-desc':        t.feat3Desc,
    'feat-4-title':       t.feat4Title,
    'feat-4-desc':        t.feat4Desc,
    'chip-1':             t.chip1,
    'chip-2':             t.chip2,
    'chip-3':             t.chip3,
    'chip-4':             t.chip4,
    'welcome-title':      t.welcomeTitle,
    'welcome-desc':       t.welcomeDesc,
    'chat-agent-name':    t.chatAgentName,
    'chat-agent-status':  t.chatAgentStatus,
    'input-hint':         t.inputHint,
    'nav-home-link':      t.navHomeLink,
    'nav-properties-link':t.navPropertiesLink,
    'nav-agencies-link':  t.navAgenciesLink,
    'nav-services-link':  t.navServicesLink,
    'nav-about-link':     t.navAboutLink,
    'nav-contact-link':   t.navContactLink,
    'nav-blog-link':      t.navBlogLink,
    'nav-login-link':     t.navLoginLink,
    'footer-agency':      t.footerAgency,
    'footer-company':     t.footerCompany,
    'footer-product':     t.footerProduct,
    'footer-areas':       t.footerAreas,
    'footer-resources':   t.footerResources,
    'ft-home':            t.navHomeLink,
    'ft-props':           t.navPropertiesLink,
    'ft-services':        t.navServicesLink,
    'ft-about':           t.navAboutLink,
    'ft-contact':         t.navContactLink,
    'ft-blog':            t.navBlogLink,
    'ft-overview':        t.ftOverview,
    'ft-pricing':         t.ftPricing,
    'ft-market':          t.ftMarket,
    'ft-features':        t.ftFeatures,
    'ft-integ':           t.ftInteg,
    'ft-cairo':           t.ftCairo,
    'ft-zayed':           t.ftZayed,
    'ft-alamein':         t.ftAlamein,
    'ft-sokhna':          t.ftSokhna,
    'ft-hekma':           t.ftHekma,
    'ft-maadi':           t.ftMaadi,
    'ft-help':            t.ftHelp,
    'ft-sales':           t.ftSales,
    'ft-ads':             t.ftAds,
    'ft-faq':             t.ftFaq,
    'footer-copyright':   t.footerCopyright,
    'ft-terms':           t.ftTerms,
    'ft-privacy':         t.ftPrivacy,
    'wc-1':               t.wc1Label,
    'wc-2':               t.wc2Label,
    'wc-3':               t.wc3Label,
    'wc-4':               t.wc4Label,
  };
  Object.entries(map).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  });

  // Also update chip data-prompt attributes
  const chips = document.querySelectorAll('#example-chips .chip');
  [t.chip1, t.chip2, t.chip3, t.chip4].forEach((txt, i) => {
    if (chips[i]) chips[i].dataset.prompt = txt;
  });

  // Welcome chips data-prompt attributes
  const wChips = [
    { el: document.getElementById('wc-1'), prompt: t.wc1Prompt },
    { el: document.getElementById('wc-2'), prompt: t.wc2Prompt },
    { el: document.getElementById('wc-3'), prompt: t.wc3Prompt },
    { el: document.getElementById('wc-4'), prompt: t.wc4Prompt },
  ];
  wChips.forEach(c => { if(c.el) c.el.dataset.prompt = c.prompt; });

  // Input placeholder
  chatInput.placeholder = t.inputPlaceholder;

  // Active lang button
  document.getElementById('btn-en').classList.toggle('active', lang === 'en');
  document.getElementById('btn-ar').classList.toggle('active', lang === 'ar');
}

/* ─────────────────────────────────────────────
   Render markdown-lite (bold, line breaks)
   ───────────────────────────────────────────── */
function renderMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,     '<em>$1</em>')
    .replace(/\n/g, '<br/>');
}

/* ─────────────────────────────────────────────
   Time helper
   ───────────────────────────────────────────── */
function nowTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/* ─────────────────────────────────────────────
   Append a message bubble
   ───────────────────────────────────────────── */
function appendMessage(role, text) {
  // Hide welcome state
  if (welcomeState) welcomeState.style.display = 'none';

  const id   = `msg-${++messageIdCounter}`;
  const time = nowTime();
  const isUser = role === 'user';

  const row = document.createElement('div');
  row.className = `message-row ${isUser ? 'user-row' : 'agent-row'}`;
  row.id = id;

  const avatarLetter = isUser ? 'U' : '🏠';
  row.innerHTML = `
    <div class="bubble-avatar" aria-hidden="true">${avatarLetter}</div>
    <div>
      <div class="bubble">${renderMarkdown(text)}</div>
      <p class="bubble-time">${time}</p>
    </div>
  `;

  messagesArea.appendChild(row);
  scrollToBottom();

  messages.push({ role, text, time });
  return id;
}

/* ─────────────────────────────────────────────
   Render Property Cards
   ───────────────────────────────────────────── */
function renderPropertyCards(properties) {
  const container = document.createElement('div');
  container.className = 'property-cards-container';
  
  properties.forEach(prop => {
    const card = document.createElement('div');
    card.className = 'property-card';
    
    const priceStr = parseFloat(prop.price_egp || 0).toLocaleString() + ' EGP';
    const titleStr = `${prop.bedrooms || '?'} Beds · ${prop.property_type || 'Property'} in ${prop.town || 'Egypt'}`;
    const descStr = prop.analyzer_reasoning || 'Selected as a strong match based on your criteria.';
    const linkStr = `https://estately.com/property/${prop.listing_id || ''}`;
    
    card.innerHTML = `
      <div class="property-card-content">
        <h4 class="property-card-price">${priceStr}</h4>
        <p class="property-card-title">${titleStr}</p>
        <div class="property-card-reasoning">
          <strong>AI Analysis:</strong> ${descStr}
        </div>
      </div>
      <a href="${linkStr}" target="_blank" class="property-card-btn">View Property</a>
    `;
    
    container.appendChild(card);
  });
  
  messagesArea.appendChild(container);
  scrollToBottom();
}

/* ─────────────────────────────────────────────
   Show / hide typing indicator
   ───────────────────────────────────────────── */
function showTyping() {
  removeTyping();
  const row = document.createElement('div');
  row.className = 'typing-row';
  row.id = 'typing-indicator';
  row.innerHTML = `
    <div class="bubble-avatar" aria-hidden="true">🏠</div>
    <div class="typing-bubble" aria-label="Agent is typing">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;
  messagesArea.appendChild(row);
  scrollToBottom();
}
function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

/* ─────────────────────────────────────────────
   Scroll to bottom
   ───────────────────────────────────────────── */
function scrollToBottom() {
  messagesArea.scrollTop = messagesArea.scrollHeight;
}

/* ─────────────────────────────────────────────
   Send message
   ───────────────────────────────────────────── */
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isTyping) return;

  // Clear input
  chatInput.value = '';
  autoResize();
  updateSendBtn();

  // Append user message
  appendMessage('user', text);

  // Show typing
  isTyping = true;
  sendBtn.disabled = true;
  showTyping();

  try {
    if (!window.estatelyThreadId) {
      try {
        window.estatelyThreadId = localStorage.getItem("estately_thread_id");
      } catch (e) {
        console.warn("localStorage not available, using RAM threadId");
      }
      if (!window.estatelyThreadId) {
        window.estatelyThreadId = "gui_" + Math.random().toString(36).substring(2, 15);
        try {
          localStorage.setItem("estately_thread_id", window.estatelyThreadId);
        } catch (e) {}
      }
    }
    
    const res = await fetch('http://127.0.0.1:8000/api/v1/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: window.estatelyThreadId, lang: currentLang })
    });
    
    const data = await res.json();
    
    removeTyping();
    if (data.reply) {
      appendMessage('agent', data.reply);
      
      if (data.recommended_properties && data.recommended_properties.length > 0) {
        renderPropertyCards(data.recommended_properties);
      }
    } else {
      appendMessage('agent', '⚠️ Error: ' + (data.error || 'Unknown error occurred.'));
    }
  } catch (err) {
    removeTyping();
    appendMessage('agent', '⚠️ Could not connect to the agent. Please ensure the API server is running on port 8000.');
  } finally {
    isTyping = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

/* ─────────────────────────────────────────────
   Use example prompt
   ───────────────────────────────────────────── */
function usePrompt(text) {
  chatInput.value = text;
  autoResize();
  updateSendBtn();
  chatInput.focus();
}

/* ─────────────────────────────────────────────
   New conversation
   ───────────────────────────────────────────── */
function newConversation() {
  messages = [];
  messageIdCounter = 0;
  isTyping = false;

  window.estatelyThreadId = null;
  try {
    localStorage.removeItem("estately_thread_id");
  } catch (e) {
    console.warn("Could not clear localStorage");
  }

  // Remove all message rows & typing
  const rows = messagesArea.querySelectorAll('.message-row, .typing-row');
  rows.forEach(r => r.remove());

  // Show welcome state again
  if (welcomeState) welcomeState.style.display = '';

  chatInput.value = '';
  autoResize();
  updateSendBtn();
  chatInput.focus();

  // Show greeting after a brief delay
  setTimeout(() => {
    appendMessage('agent', i18n[currentLang].greetingMsg);
  }, 400);
}

/* ─────────────────────────────────────────────
   Auto-resize textarea
   ───────────────────────────────────────────── */
function autoResize() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

/* ─────────────────────────────────────────────
   Update send button state
   ───────────────────────────────────────────── */
function updateSendBtn() {
  const hasText = chatInput.value.trim().length > 0;
  sendBtn.disabled = !hasText || isTyping;
  charCount.textContent = `${chatInput.value.length} / 500`;
}

/* ─────────────────────────────────────────────
   Download chat as .txt
   ───────────────────────────────────────────── */
const downloadBtn = document.getElementById('download-chat-btn');
if (downloadBtn) {
  downloadBtn.addEventListener('click', () => {
    if (!messages.length) return;
    const lines = messages.map(m => `[${m.time}] ${m.role === 'user' ? 'You' : 'Estately AI'}: ${m.text}`);
    const blob = new Blob([lines.join('\n\n')], { type: 'text/plain' });
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(blob),
      download: `estately-chat-${Date.now()}.txt`
    });
    a.click();
  });
}

/* ─────────────────────────────────────────────
   Event listeners
   ───────────────────────────────────────────── */
chatInput.addEventListener('input', () => {
  autoResize();
  updateSendBtn();
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

/* ─────────────────────────────────────────────
   Init
   ───────────────────────────────────────────── */
(function init() {
  setLang('en');  // default language
  updateSendBtn();
  chatInput.focus();

  // Show greeting after a brief delay
  setTimeout(() => {
    appendMessage('agent', i18n[currentLang].greetingMsg);
  }, 400);
})();
