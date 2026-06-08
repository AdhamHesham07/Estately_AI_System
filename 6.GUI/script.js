/* =============================================
   ESTATELY — Homepage JavaScript
   Features: Scroll effects, counter animation,
             scroll-reveal, nav highlight, fav toggle
   ============================================= */

document.addEventListener('DOMContentLoaded', () => {

  /* ─── 1. NAVBAR: shadow on scroll ─── */
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
  });

  /* ─── 2. SCROLL-REVEAL (fade-up elements) ─── */
  const fadeEls = document.querySelectorAll('.fade-up');
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  fadeEls.forEach(el => revealObserver.observe(el));

  /* ─── 3. STATS COUNTER ANIMATION ─── */
  const statsSection = document.getElementById('stats');
  let counted = false;

  const countObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && !counted) {
      counted = true;
      animateCounters();
    }
  }, { threshold: 0.4 });

  if (statsSection) countObserver.observe(statsSection);

  function animateCounters() {
    const counters = document.querySelectorAll('.stat-number[data-target]');
    counters.forEach(counter => {
      const target = parseInt(counter.dataset.target, 10);
      const duration = 1800;
      const step = 16; // ~60fps
      const totalSteps = duration / step;
      let current = 0;

      const timer = setInterval(() => {
        current += target / totalSteps;
        if (current >= target) {
          current = target;
          clearInterval(timer);
        }
        counter.textContent = Math.floor(current).toLocaleString();
      }, step);
    });
  }

  /* ─── 4. FAVOURITE BUTTON TOGGLE ─── */
  document.querySelectorAll('.property-fav').forEach(btn => {
    btn.addEventListener('click', () => {
      const isFav = btn.textContent.trim() === '♥';
      btn.textContent = isFav ? '♡' : '♥';
      btn.style.color = isFav ? '#ef4444' : '#ef4444';
    });
  });

  /* ─── 5. SMOOTH SCROLL for nav links ─── */
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      const targetId = link.getAttribute('href');
      if (targetId === '#') return;
      const target = document.querySelector(targetId);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  /* ─── 6. ACTIVE NAV LINK on scroll ─── */
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a');

  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => link.classList.remove('active'));
        const id = entry.target.getAttribute('id');
        const match = document.querySelector(`.nav-links a[href="#${id}"]`);
        if (match) match.classList.add('active');
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(s => sectionObserver.observe(s));

  /* ─── 7. SEARCH BUTTON feedback ─── */
  const searchBtn = document.getElementById('search-btn');
  if (searchBtn) {
    searchBtn.addEventListener('click', () => {
      searchBtn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Searching…';
      setTimeout(() => {
        searchBtn.innerHTML = '<i class="bx bx-search-alt"></i> Search';
      }, 1800);
    });
  }

  /* ─── 8. SECTION NAV ARROWS (placeholder feedback) ─── */
  ['prev-btn', 'next-btn'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener('click', () => {
        btn.style.transform = 'scale(0.9)';
        setTimeout(() => btn.style.transform = '', 200);
      });
    }
  });

});
