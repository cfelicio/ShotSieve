(() => {
  function createWorkflowLibraryBrowser(deps) {
    const {
      api,
      formatting,
      notifications,
      review,
      state,
      ui,
    } = deps;

    const { fetchJson } = api;
    const { escapeHtml } = formatting;
    const { showToast } = notifications;
    const { loadQueue, selectFile } = review;
    const { currentLibraryRoot } = ui;

    function renderLibraryRoots() {
      const listContainer = document.getElementById("library-roots-list");
      const rootStr = currentLibraryRoot();
      const roots = rootStr.split("|").map((r) => r.trim()).filter(Boolean);

      const decisionRoot = document.getElementById("decision-csv-root");
      if (decisionRoot) {
        const previousRoot = decisionRoot.value;
        decisionRoot.replaceChildren(new Option("Choose a library root", ""));
        roots.forEach((root) => decisionRoot.add(new Option(root, root)));
        if (roots.includes(previousRoot)) {
          decisionRoot.value = previousRoot;
        } else if (roots.length === 1) {
          decisionRoot.value = roots[0];
        }
      }

      if (!listContainer) return;

      if (roots.length === 0) {
        listContainer.innerHTML = `<p class="muted">No folders selected yet. Click "Add Folder" to add directories to your library.</p>`;
        return;
      }

      listContainer.innerHTML = roots.map((rootPath) => `
        <div class="library-root-item">
          <span class="library-root-path">${escapeHtml(rootPath)}</span>
          <button type="button" class="library-root-remove" data-path="${escapeHtml(rootPath)}" aria-label="Remove folder">✕</button>
        </div>
      `).join("");

      listContainer.querySelectorAll(".library-root-remove").forEach((button) => {
        button.addEventListener("click", () => {
          const pathToRemove = button.dataset.path;
          const updatedRoots = roots.filter((root) => root !== pathToRemove);
          const hiddenInput = document.getElementById("library-root-input");
          if (hiddenInput) {
            hiddenInput.value = updatedRoots.join("|");
            hiddenInput.dispatchEvent(new Event("input", { bubbles: true }));
            hiddenInput.dispatchEvent(new Event("change", { bubbles: true }));
            renderLibraryRoots();
          }
        });
      });
    }

    async function downloadDecisionCsv() {
      const root = document.getElementById("decision-csv-root")?.value || "";
      const decision = document.getElementById("decision-csv-decision")?.value || "both";
      if (!root) {
        throw new Error("Choose a library root before downloading decisions.");
      }
      const response = await fetch(`/api/review/decisions.csv?root=${encodeURIComponent(root)}&decision=${encodeURIComponent(decision)}`);
      if (!response.ok) {
        let message = `Decision CSV request failed (${response.status}).`;
        try {
          const payload = await response.json();
          message = payload?.error || message;
        } catch {
          // Keep the status-based message for non-JSON server errors.
        }
        throw new Error(message);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "shotsieve-decisions.csv";
      link.click();
      URL.revokeObjectURL(url);
      showToast("Decision CSV downloaded.");
    }

    function installDecisionCsvEvents() {
      const button = document.getElementById("download-decisions-csv");
      if (!button || button.dataset.eventsInstalled === "true") return;
      button.dataset.eventsInstalled = "true";
      button.addEventListener("click", () => {
        downloadDecisionCsv().catch(handleError);
      });
    }

    async function navigateSelection(step) {
      if (!state.queue.length) return;
      const currentIndex = state.queue.findIndex((item) => item.id === state.activeId);
      if (currentIndex === -1) {
        await selectFile(state.queue[0].id);
        return;
      }

      const nextIndex = currentIndex + step;
      if (nextIndex >= 0 && nextIndex < state.queue.length) {
        await selectFile(state.queue[nextIndex].id);
        return;
      }

      if (step > 0 && ((state.page + 1) * state.pageSize) < state.totalFiles) {
        state.page += 1;
        await loadQueue();
        if (state.queue.length) {
          await selectFile(state.queue[0].id);
        }
        return;
      }

      if (step < 0 && state.page > 0) {
        state.page -= 1;
        await loadQueue();
        if (state.queue.length) {
          await selectFile(state.queue[state.queue.length - 1].id);
        }
      }
    }

    async function openOriginalFile(fileId) {
      if (!Number.isInteger(Number(fileId)) || Number(fileId) <= 0) {
        throw new Error("Pick a file first.");
      }
      await deps.api.postJson("/api/files/open", { file_id: Number(fileId) });
    }

    async function openBrowser(targetId) {
      state.browserTarget = targetId;
      const dialog = document.getElementById("folder-browser");
      if (!dialog.open) {
        dialog.showModal();
      }
      const roots = await fetchJson("/api/fs/roots");
      const rootContainer = document.getElementById("browser-roots");
      rootContainer.innerHTML = roots.items.map((item) => `<button type="button" class="ghost browser-root" data-path="${escapeHtml(item.path)}">${escapeHtml(item.name)}</button>`).join("");
      rootContainer.querySelectorAll(".browser-root").forEach((button) => {
        button.addEventListener("click", () => browseDirectory(button.dataset.path).catch(handleError));
      });

      let startPath = "";
      const targetEl = document.getElementById(targetId);
      if (targetEl && targetEl.value) {
        startPath = targetEl.value;
      } else {
        const currentLibraryVal = document.getElementById("library-root-input")?.value;
        if (currentLibraryVal) {
          const libraryRoots = currentLibraryVal.split("|").map((r) => r.trim()).filter(Boolean);
          if (libraryRoots.length > 0) {
            startPath = libraryRoots[libraryRoots.length - 1];
          }
        }
      }
      if (!startPath) {
        startPath = state.browserPath || roots.items[0]?.path || "/";
      }

      try {
        await browseDirectory(startPath);
      } catch (err) {
        console.warn("Failed to navigate to browser start path, falling back to root:", err);
        const fallback = roots.items[0]?.path || "/";
        await browseDirectory(fallback).catch(handleError);
      }
    }

    function buildBreadcrumbItems(rawPath) {
      const isUnc = rawPath.startsWith("\\\\") || rawPath.startsWith("//");
      const normPath = rawPath.replace(/\\/g, "/");

      if (isUnc) {
        const parts = normPath.slice(2).split("/").filter(Boolean);
        let accumulated = "\\\\";
        return parts.map((part, index) => {
          if (index === 0) {
            accumulated += part;
          } else {
            accumulated += "\\" + part;
          }
          return `<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(part)}</button>`;
        });
      }

      const isWindowsDrive = /^[a-zA-Z]:/.test(normPath);
      if (isWindowsDrive) {
        const driveLetter = normPath.slice(0, 2);
        const rest = normPath.slice(2).split("/").filter(Boolean);
        let accumulated = `${driveLetter}\\`;
        const crumbs = [
          `<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(driveLetter)}</button>`,
        ];
        for (const part of rest) {
          accumulated += (accumulated.endsWith("\\") ? "" : "\\") + part;
          crumbs.push(`<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(part)}</button>`);
        }
        return crumbs;
      }

      const parts = normPath.split("/").filter(Boolean);
      let accumulated = "/";
      const crumbs = [
        `<button type="button" class="breadcrumb-item" data-path="/">${escapeHtml("/")}</button>`,
      ];
      for (const part of parts) {
        accumulated += (accumulated.endsWith("/") ? "" : "/") + part;
        crumbs.push(`<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(part)}</button>`);
      }
      return crumbs;
    }

    let activeBrowseSeq = 0;

    async function browseDirectory(path) {
      const currentSeq = ++activeBrowseSeq;
      const list = document.getElementById("browser-list");
      const pathInput = document.getElementById("browser-path");
      state.browserPath = null;
      if (pathInput) pathInput.value = path;

      if (list && !list.children.length) {
        list.innerHTML = `<p class="muted">Loading directory contents...</p>`;
      }

      try {
        const payload = await fetchJson(`/api/fs/list?path=${encodeURIComponent(path)}`);
        if (currentSeq !== activeBrowseSeq) {
          return;
        }

        state.browserPath = payload.path;
        if (pathInput) pathInput.value = payload.path;

        if (list) {
          list.innerHTML = payload.items.length
            ? payload.items.map((item) => `
                <button type="button" class="browser-item" data-path="${escapeHtml(item.path)}">
                  <strong>${escapeHtml(item.name)}</strong>
                  <span class="muted">${escapeHtml(item.path)}</span>
                </button>
              `).join("")
            : `<p class="muted">No subdirectories available.</p>`;

          list.querySelectorAll(".browser-item").forEach((button) => {
            button.addEventListener("click", () => browseDirectory(button.dataset.path).catch(handleError));
          });
        }

        const breadcrumbsContainer = document.getElementById("browser-breadcrumbs");
        if (breadcrumbsContainer) {
          breadcrumbsContainer.innerHTML = buildBreadcrumbItems(payload.path).join('<span class="breadcrumb-separator">/</span>');
          breadcrumbsContainer.querySelectorAll(".breadcrumb-item").forEach((button) => {
            button.addEventListener("click", () => browseDirectory(button.dataset.path).catch(handleError));
          });
        }
      } catch (err) {
        if (currentSeq !== activeBrowseSeq) {
          return;
        }
        if (list) {
          list.innerHTML = `<p class="muted danger-text">Could not open folder: ${escapeHtml(err.message || "Access denied")}</p>`;
        }
      }
    }

    function chooseBrowserPath() {
      if (!state.browserTarget) return;
      const selectedPath = state.browserPath || document.getElementById("browser-path")?.value?.trim();
      if (!selectedPath) return;
      const targetInput = document.getElementById(state.browserTarget);
      if (!targetInput) {
        return;
      }
      targetInput.value = selectedPath;
      targetInput.dispatchEvent(new Event("input", { bubbles: true }));
      targetInput.dispatchEvent(new Event("change", { bubbles: true }));
      document.getElementById("folder-browser").close();
    }

    function handleError(error) {
      console.error(error);
      let message = error?.message || "Unexpected error";
      if (message === "Failed to fetch") {
        message = "The local server request failed. If an analysis is still running, wait for completion before retrying.";
      }
      showToast(message, "error");
    }

    return {
      renderLibraryRoots,
      downloadDecisionCsv,
      installDecisionCsvEvents,
      navigateSelection,
      openOriginalFile,
      openBrowser,
      browseDirectory,
      chooseBrowserPath,
      handleError,
    };
  }

  window.ShotSieveWorkflowLibraryBrowser = {
    createWorkflowLibraryBrowser,
  };
})();
