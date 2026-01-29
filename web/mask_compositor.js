import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Mask Compositor Widget
// Allows painting/editing masks over images with a mini canvas interface

app.registerExtension({
    name: "TextureAlchemy.MaskCompositor",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MaskCompositor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Add hidden widget to store mask data
                this.addWidget("string", "mask_data", "", () => {}, {
                    serialize: true
                });
                
                // Add button to open compositor
                this.addWidget("button", "Open Mask Editor", null, () => {
                    this.openMaskCompositor();
                });
                
                return r;
            };
            
            // Get image data from inputs
            nodeType.prototype.getInputImageData = async function() {
                // Get the image input
                const imageInput = this.inputs[0];
                if (!imageInput || !imageInput.link) {
                    return null;
                }
                
                // Get the node that's connected
                const link = this.graph.links[imageInput.link];
                if (!link) return null;
                
                const originNode = this.graph.getNodeById(link.origin_id);
                if (!originNode) return null;
                
                // Get image from origin node's images property
                if (originNode.imgs && originNode.imgs.length > 0) {
                    return originNode.imgs[0];
                }
                
                return null;
            };
            
            // Open compositor in a dialog
            nodeType.prototype.openMaskCompositor = async function() {
                // Get current image
                const imgElement = await this.getInputImageData();
                
                // Create compositor dialog
                const dialog = document.createElement("div");
                dialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #1e1e1e;
                    border: 2px solid #444;
                    border-radius: 8px;
                    padding: 20px;
                    z-index: 10000;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    max-width: 90vw;
                    max-height: 90vh;
                `;
                
                // Create header
                const header = document.createElement("div");
                header.style.cssText = `
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    color: #fff;
                    font-family: Arial, sans-serif;
                `;
                header.innerHTML = `
                    <h3 style="margin: 0;">Mask Compositor</h3>
                    <button id="closeCompositor" style="
                        background: #ff4444;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 5px 15px;
                        cursor: pointer;
                    ">Close</button>
                `;
                
                // Determine canvas size
                let canvasWidth = 512;
                let canvasHeight = 512;
                
                if (imgElement) {
                    // Use image dimensions
                    canvasWidth = Math.min(imgElement.naturalWidth || imgElement.width, 1024);
                    canvasHeight = Math.min(imgElement.naturalHeight || imgElement.height, 1024);
                }
                
                // Create canvas container
                const canvasContainer = document.createElement("div");
                canvasContainer.style.cssText = `
                    position: relative;
                    background: #000;
                    border: 1px solid #555;
                    margin-bottom: 15px;
                `;
                
                // Create canvas for rendering
                const canvas = document.createElement("canvas");
                canvas.width = canvasWidth;
                canvas.height = canvasHeight;
                canvas.style.cssText = `
                    display: block;
                    cursor: crosshair;
                    max-width: 100%;
                    height: auto;
                `;
                
                // Create tools
                const tools = document.createElement("div");
                tools.style.cssText = `
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                    align-items: center;
                `;
                tools.innerHTML = `
                    <button id="toolPaint" style="background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold;">Paint</button>
                    <button id="toolErase" style="background: #f44336; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Erase</button>
                    <label style="color: #fff; display: flex; align-items: center; gap: 5px;">
                        Shape:
                        <select id="brushShape" style="padding: 4px; border-radius: 4px; background: #333; color: white; border: 1px solid #555;">
                            <option value="circle">Circle</option>
                            <option value="square">Square</option>
                            <option value="soft">Soft Circle</option>
                        </select>
                    </label>
                    <label style="color: #fff; display: flex; align-items: center; gap: 5px;">
                        Size: 
                        <input type="range" id="brushSize" min="1" max="100" value="20" style="width: 100px;">
                        <span id="brushSizeValue">20</span>
                    </label>
                    <label style="color: #fff; display: flex; align-items: center; gap: 5px;">
                        Hardness: 
                        <input type="range" id="brushHardness" min="0" max="100" value="100" style="width: 80px;">
                        <span id="brushHardnessValue">100%</span>
                    </label>
                    <label style="color: #fff; display: flex; align-items: center; gap: 5px;">
                        Opacity: 
                        <input type="range" id="maskOpacity" min="0" max="100" value="70" style="width: 80px;">
                        <span id="maskOpacityValue">70%</span>
                    </label>
                    <button id="clearMask" style="background: #ff9800; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Clear</button>
                    <button id="invertMask" style="background: #2196F3; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Invert</button>
                    <button id="saveMask" style="background: #9C27B0; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold;">Save to Node</button>
                `;
                
                // Assemble dialog
                canvasContainer.appendChild(canvas);
                dialog.appendChild(header);
                dialog.appendChild(canvasContainer);
                dialog.appendChild(tools);
                document.body.appendChild(dialog);
                
                // Initialize compositor
                const ctx = canvas.getContext("2d", { willReadFrequently: true });
                let currentTool = "paint";
                let brushSize = 20;
                let brushShape = "circle";
                let brushHardness = 1.0;
                let maskOpacity = 0.7;
                let isDrawing = false;
                let lastX = -1;
                let lastY = -1;
                
                // Load existing mask data if available
                const maskDataWidget = this.widgets.find(w => w.name === "mask_data");
                let maskData = ctx.createImageData(canvas.width, canvas.height);
                
                if (maskDataWidget && maskDataWidget.value) {
                    try {
                        const savedData = JSON.parse(maskDataWidget.value);
                        if (savedData.width === canvas.width && savedData.height === canvas.height) {
                            // Load saved mask
                            const uint8Array = new Uint8ClampedArray(savedData.data);
                            maskData = new ImageData(uint8Array, canvas.width, canvas.height);
                            console.log("Loaded existing mask data");
                        }
                    } catch (e) {
                        console.log("No valid mask data to load, starting fresh");
                    }
                }
                
                // Initialize mask as transparent if new
                if (!maskDataWidget || !maskDataWidget.value) {
                    for (let i = 0; i < maskData.data.length; i += 4) {
                        maskData.data[i] = 0;     // R
                        maskData.data[i + 1] = 0; // G
                        maskData.data[i + 2] = 0; // B
                        maskData.data[i + 3] = 0; // A
                    }
                }
                
                // Render function
                const render = () => {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw image if available
                    if (imgElement) {
                        ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
                    } else {
                        // Placeholder
                        ctx.fillStyle = "#333";
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = "#fff";
                        ctx.font = "16px Arial";
                        ctx.textAlign = "center";
                        ctx.fillText("Connect image to see preview", canvas.width / 2, canvas.height / 2);
                    }
                    
                    // Draw mask overlay
                    const tempCanvas = document.createElement("canvas");
                    tempCanvas.width = canvas.width;
                    tempCanvas.height = canvas.height;
                    const tempCtx = tempCanvas.getContext("2d");
                    tempCtx.putImageData(maskData, 0, 0);
                    
                    ctx.globalAlpha = maskOpacity;
                    ctx.drawImage(tempCanvas, 0, 0);
                    ctx.globalAlpha = 1.0;
                };
                
                // Drawing functions - improved with shapes and smoothness
                const drawOnMask = (x, y) => {
                    const color = currentTool === "paint" ? 255 : 0;
                    const radius = brushSize;
                    
                    for (let dy = -radius; dy <= radius; dy++) {
                        for (let dx = -radius; dx <= radius; dx++) {
                            const px = Math.floor(x + dx);
                            const py = Math.floor(y + dy);
                            
                            if (px >= 0 && px < canvas.width && py >= 0 && py < canvas.height) {
                                let alpha = 1.0;
                                
                                if (brushShape === "circle" || brushShape === "soft") {
                                    const dist = Math.sqrt(dx * dx + dy * dy);
                                    if (dist > radius) continue;
                                    
                                    if (brushShape === "soft") {
                                        // Soft falloff
                                        alpha = Math.max(0, 1 - (dist / radius));
                                    } else {
                                        // Hard circle with edge smoothing
                                        if (dist > radius - 1) {
                                            alpha = radius - dist; // Smooth edge
                                        }
                                    }
                                } else if (brushShape === "square") {
                                    // Hard square
                                    if (Math.abs(dx) > radius || Math.abs(dy) > radius) continue;
                                    
                                    // Optional: soft edges on square
                                    const edgeDist = Math.max(
                                        Math.abs(dx) - (radius - 1),
                                        Math.abs(dy) - (radius - 1)
                                    );
                                    if (edgeDist > 0) {
                                        alpha = 1 - edgeDist;
                                    }
                                }
                                
                                // Apply hardness
                                if (brushHardness < 1.0) {
                                    alpha = Math.pow(alpha, 1 / Math.max(0.01, brushHardness));
                                }
                                
                                if (alpha > 0) {
                                    const idx = (py * canvas.width + px) * 4;
                                    const finalAlpha = Math.floor(color * alpha);
                                    
                                    // Blend with existing
                                    if (currentTool === "paint") {
                                        maskData.data[idx] = Math.max(maskData.data[idx], color);
                                        maskData.data[idx + 1] = Math.max(maskData.data[idx + 1], color);
                                        maskData.data[idx + 2] = Math.max(maskData.data[idx + 2], color);
                                        maskData.data[idx + 3] = Math.max(maskData.data[idx + 3], finalAlpha);
                                    } else {
                                        // Erase
                                        const eraseStrength = alpha;
                                        maskData.data[idx] = Math.floor(maskData.data[idx] * (1 - eraseStrength));
                                        maskData.data[idx + 1] = Math.floor(maskData.data[idx + 1] * (1 - eraseStrength));
                                        maskData.data[idx + 2] = Math.floor(maskData.data[idx + 2] * (1 - eraseStrength));
                                        maskData.data[idx + 3] = Math.floor(maskData.data[idx + 3] * (1 - eraseStrength));
                                    }
                                }
                            }
                        }
                    }
                };
                
                // Draw smooth line between points for continuous strokes
                const drawLine = (x0, y0, x1, y1) => {
                    const dist = Math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2);
                    const steps = Math.max(1, Math.ceil(dist / (brushSize * 0.25)));
                    
                    for (let i = 0; i <= steps; i++) {
                        const t = i / steps;
                        const x = x0 + (x1 - x0) * t;
                        const y = y0 + (y1 - y0) * t;
                        drawOnMask(x, y);
                    }
                };
                
                // Mouse events
                const getMousePos = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    return {
                        x: (e.clientX - rect.left) * scaleX,
                        y: (e.clientY - rect.top) * scaleY
                    };
                };
                
                canvas.addEventListener("mousedown", (e) => {
                    isDrawing = true;
                    const pos = getMousePos(e);
                    lastX = pos.x;
                    lastY = pos.y;
                    drawOnMask(pos.x, pos.y);
                    render();
                });
                
                canvas.addEventListener("mousemove", (e) => {
                    if (isDrawing) {
                        const pos = getMousePos(e);
                        
                        // Draw smooth line from last position
                        if (lastX !== -1 && lastY !== -1) {
                            drawLine(lastX, lastY, pos.x, pos.y);
                        } else {
                            drawOnMask(pos.x, pos.y);
                        }
                        
                        lastX = pos.x;
                        lastY = pos.y;
                        render();
                    }
                });
                
                canvas.addEventListener("mouseup", () => {
                    isDrawing = false;
                    lastX = -1;
                    lastY = -1;
                });
                
                canvas.addEventListener("mouseleave", () => {
                    isDrawing = false;
                    lastX = -1;
                    lastY = -1;
                });
                
                // Tool buttons
                const paintBtn = dialog.querySelector("#toolPaint");
                const eraseBtn = dialog.querySelector("#toolErase");
                
                paintBtn.addEventListener("click", () => {
                    currentTool = "paint";
                    paintBtn.style.fontWeight = "bold";
                    eraseBtn.style.fontWeight = "normal";
                });
                
                eraseBtn.addEventListener("click", () => {
                    currentTool = "erase";
                    eraseBtn.style.fontWeight = "bold";
                    paintBtn.style.fontWeight = "normal";
                });
                
                dialog.querySelector("#brushShape").addEventListener("change", (e) => {
                    brushShape = e.target.value;
                });
                
                dialog.querySelector("#brushSize").addEventListener("input", (e) => {
                    brushSize = parseInt(e.target.value);
                    dialog.querySelector("#brushSizeValue").textContent = brushSize;
                });
                
                dialog.querySelector("#brushHardness").addEventListener("input", (e) => {
                    brushHardness = parseInt(e.target.value) / 100;
                    dialog.querySelector("#brushHardnessValue").textContent = e.target.value + "%";
                });
                
                dialog.querySelector("#maskOpacity").addEventListener("input", (e) => {
                    maskOpacity = parseInt(e.target.value) / 100;
                    dialog.querySelector("#maskOpacityValue").textContent = e.target.value + "%";
                    render();
                });
                
                dialog.querySelector("#clearMask").addEventListener("click", () => {
                    for (let i = 0; i < maskData.data.length; i++) {
                        maskData.data[i] = 0;
                    }
                    render();
                });
                
                dialog.querySelector("#invertMask").addEventListener("click", () => {
                    for (let i = 0; i < maskData.data.length; i += 4) {
                        maskData.data[i] = 255 - maskData.data[i];
                        maskData.data[i + 1] = 255 - maskData.data[i + 1];
                        maskData.data[i + 2] = 255 - maskData.data[i + 2];
                    }
                    render();
                });
                
                dialog.querySelector("#saveMask").addEventListener("click", () => {
                    // Save mask data to widget
                    const maskDataWidget = this.widgets.find(w => w.name === "mask_data");
                    if (maskDataWidget) {
                        const savedData = {
                            width: canvas.width,
                            height: canvas.height,
                            data: Array.from(maskData.data)
                        };
                        maskDataWidget.value = JSON.stringify(savedData);
                        console.log("Mask saved to node!");
                        alert("✓ Mask saved to node! Close editor and run workflow.");
                    }
                });
                
                dialog.querySelector("#closeCompositor").addEventListener("click", () => {
                    document.body.removeChild(dialog);
                });
                
                // Initial render
                render();
            };
        }
    }
});
