/**
 * Color Picker Widget for ColorImage Node
 * Provides an interactive color wheel and hex input
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "TextureAlchemy.ColorPickerWidget",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ColorImage" || nodeData.name === "ColorCode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                const colorWidget = this.widgets.find(w => w.name === "color");
                
                if (colorWidget) {
                    // Ensure alpha widget exists (create if hidden)
                    let alphaWidget = this.widgets.find(w => w.name === "alpha");
                    if (!alphaWidget) {
                        // Create hidden alpha widget if it doesn't exist
                        alphaWidget = this.addWidget("number", "alpha", 1.0, () => {}, { 
                            min: 0.0, 
                            max: 1.0, 
                            step: 0.01,
                            precision: 2
                        });
                        // Hide it from view
                        alphaWidget.computeSize = () => [0, -4];
                    }
                    
                    // Store original callback
                    const originalCallback = colorWidget.callback;
                    
                    // Add color picker button
                    this.addWidget(
                        "button",
                        "🎨 Pick Color",
                        null,
                        () => {
                            this.openColorPicker(colorWidget);
                        }
                    );
                    
                    // Add eyedropper button (if supported)
                    if (window.EyeDropper) {
                        this.addWidget(
                            "button",
                            "💧 Eyedropper",
                            null,
                            async () => {
                                try {
                                    const eyeDropper = new EyeDropper();
                                    const result = await eyeDropper.open();
                                    colorWidget.value = result.sRGBHex.toUpperCase();
                                    if (colorWidget.callback) {
                                        colorWidget.callback(colorWidget.value);
                                    }
                                } catch (err) {
                                    console.log("Eyedropper cancelled or failed:", err);
                                }
                            }
                        );
                    } else {
                        // Fallback button that explains browser doesn't support it
                        this.addWidget(
                            "button",
                            "💧 Eyedropper (Not Supported)",
                            null,
                            () => {
                                alert("EyeDropper API not supported in this browser.\nTry Chrome/Edge 95+ or Safari 16.4+");
                            }
                        );
                    }
                    
                    // Update preview color whenever color changes
                    const originalWidgetCallback = colorWidget.callback;
                    colorWidget.callback = function(value) {
                        if (originalWidgetCallback) originalWidgetCallback.call(this, value);
                        updateColorPreview(this.node, value);
                    };
                    
                    // Initial preview update
                    updateColorPreview(this, colorWidget.value);
                    
                    // Add extra space at bottom for color bar
                    this.size[1] = this.computeSize()[1] + 40;
                }
                
                return result;
            };
            
            // Color picker dialog
            nodeType.prototype.openColorPicker = function(colorWidget) {
                const alphaWidget = this.widgets.find(w => w.name === "alpha");
                const currentAlpha = alphaWidget ? alphaWidget.value : 1.0;
                
                const dialog = document.createElement("div");
                dialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #222;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    z-index: 10000;
                    color: white;
                    font-family: Arial, sans-serif;
                `;
                
                dialog.innerHTML = `
                    <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold;">
                        🎨 Color Picker
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 15px;">
                        <div>
                            <label style="display: block; margin-bottom: 5px; font-size: 12px;">
                                Color Wheel:
                            </label>
                            <input type="color" id="colorWheel" value="${colorWidget.value}" 
                                   style="width: 100%; height: 60px; cursor: pointer; border: none; border-radius: 4px;">
                        </div>
                        <div>
                            <label style="display: block; margin-bottom: 5px; font-size: 12px;">
                                Hex Code:
                            </label>
                            <div style="display: flex; gap: 8px;">
                                <input type="text" id="colorHex" value="${colorWidget.value}" 
                                       style="flex: 1; padding: 8px; background: #333; border: 1px solid #555; 
                                              color: white; border-radius: 4px; font-family: monospace;">
                                ${window.EyeDropper ? `
                                <button id="eyedropperBtn" title="Pick color from screen" 
                                        style="padding: 8px 12px; background: #555; color: white; border: none; 
                                               border-radius: 4px; cursor: pointer; font-size: 16px;">
                                    💧
                                </button>
                                ` : ''}
                            </div>
                        </div>
                        ${alphaWidget ? `
                        <div>
                            <label style="display: block; margin-bottom: 5px; font-size: 12px;">
                                Alpha: <span id="alphaValue">${(currentAlpha * 100).toFixed(0)}%</span>
                            </label>
                            <input type="range" id="alphaSlider" min="0" max="100" value="${currentAlpha * 100}" 
                                   style="width: 100%; height: 8px; cursor: pointer; accent-color: #4CAF50;">
                        </div>
                        ` : ''}
                        <div>
                            <label style="display: block; margin-bottom: 5px; font-size: 12px;">
                                Preview:
                            </label>
                            <div id="colorPreview" style="width: 100%; height: 40px; border-radius: 4px; 
                                                          border: 2px solid #444; background: ${colorWidget.value};">
                            </div>
                        </div>
                        <div style="display: flex; gap: 10px; margin-top: 10px;">
                            <button id="colorOk" style="flex: 1; padding: 10px; background: #4CAF50; 
                                                        color: white; border: none; border-radius: 4px; 
                                                        cursor: pointer; font-size: 14px; font-weight: bold;">
                                ✓ OK
                            </button>
                            <button id="colorCancel" style="flex: 1; padding: 10px; background: #666; 
                                                            color: white; border: none; border-radius: 4px; 
                                                            cursor: pointer; font-size: 14px;">
                                Cancel
                            </button>
                        </div>
                        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
                            <label style="display: block; margin-bottom: 8px; font-size: 12px; color: #aaa;">
                                Quick Colors:
                            </label>
                            <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 5px;">
                                ${generateQuickColors()}
                            </div>
                        </div>
                    </div>
                `;
                
                // Add overlay
                const overlay = document.createElement("div");
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0,0,0,0.7);
                    z-index: 9999;
                `;
                
                document.body.appendChild(overlay);
                document.body.appendChild(dialog);
                
                // Get elements
                const colorWheel = dialog.querySelector("#colorWheel");
                const colorHex = dialog.querySelector("#colorHex");
                const colorPreview = dialog.querySelector("#colorPreview");
                const alphaSlider = dialog.querySelector("#alphaSlider");
                const alphaValue = dialog.querySelector("#alphaValue");
                const okButton = dialog.querySelector("#colorOk");
                const cancelButton = dialog.querySelector("#colorCancel");
                const eyedropperBtn = dialog.querySelector("#eyedropperBtn");
                const quickColors = dialog.querySelectorAll(".quick-color");
                
                // Update functions
                const updatePreview = (color, alpha) => {
                    const a = alpha !== undefined ? alpha : (alphaSlider ? alphaSlider.value / 100 : 1.0);
                    colorPreview.style.background = color;
                    colorHex.value = color.toUpperCase();
                    colorWheel.value = color;
                    
                    // Update alpha display
                    if (alphaValue && alphaSlider) {
                        alphaValue.textContent = `${Math.round(a * 100)}%`;
                    }
                };
                
                // Event listeners
                colorWheel.addEventListener("input", (e) => {
                    updatePreview(e.target.value);
                });
                
                colorHex.addEventListener("input", (e) => {
                    let value = e.target.value.trim();
                    if (!value.startsWith('#')) value = '#' + value;
                    if (/^#[0-9A-Fa-f]{6}$/.test(value)) {
                        updatePreview(value);
                    }
                });
                
                // Alpha slider
                if (alphaSlider) {
                    alphaSlider.addEventListener("input", (e) => {
                        const alpha = e.target.value / 100;
                        updatePreview(colorHex.value, alpha);
                    });
                }
                
                // Eyedropper button in dialog
                if (eyedropperBtn) {
                    eyedropperBtn.addEventListener("click", async () => {
                        try {
                            const eyeDropper = new EyeDropper();
                            const result = await eyeDropper.open();
                            updatePreview(result.sRGBHex.toUpperCase());
                        } catch (err) {
                            console.log("Eyedropper cancelled:", err);
                        }
                    });
                }
                
                quickColors.forEach(btn => {
                    btn.addEventListener("click", (e) => {
                        updatePreview(e.target.dataset.color);
                    });
                });
                
                okButton.addEventListener("click", () => {
                    colorWidget.value = colorHex.value.toUpperCase();
                    if (colorWidget.callback) {
                        colorWidget.callback(colorWidget.value);
                    }
                    
                    // Update alpha widget if it exists
                    if (alphaWidget && alphaSlider) {
                        alphaWidget.value = alphaSlider.value / 100;
                        if (alphaWidget.callback) {
                            alphaWidget.callback(alphaWidget.value);
                        }
                    }
                    
                    document.body.removeChild(dialog);
                    document.body.removeChild(overlay);
                });
                
                cancelButton.addEventListener("click", () => {
                    document.body.removeChild(dialog);
                    document.body.removeChild(overlay);
                });
                
                overlay.addEventListener("click", () => {
                    document.body.removeChild(dialog);
                    document.body.removeChild(overlay);
                });
                
                // Focus hex input
                colorHex.focus();
                colorHex.select();
            };
        }
    }
});

// Helper: Update color preview widget
function updateColorPreview(node, color) {
    const previewWidget = node.widgets.find(w => w.name === "Color Preview");
    if (previewWidget) {
        // Store color for custom rendering
        node._previewColor = color;
        node.setDirtyCanvas(true, true);
    }
}

// Helper: Generate quick color palette
function generateQuickColors() {
    const colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
        "#FF00FF", "#00FFFF", "#FFFFFF", "#000000",
        "#FF8800", "#88FF00", "#0088FF", "#8800FF",
        "#FF0088", "#00FF88", "#888888", "#444444",
    ];
    
    return colors.map(color => 
        `<button class="quick-color" data-color="${color}" 
                style="width: 100%; aspect-ratio: 1; background: ${color}; 
                       border: 1px solid #666; border-radius: 3px; cursor: pointer;
                       transition: transform 0.1s;"
                onmouseover="this.style.transform='scale(1.1)'"
                onmouseout="this.style.transform='scale(1)'"
         ></button>`
    ).join('');
}

// Custom node rendering to show color preview
const originalDrawNodeShape = LGraphCanvas.prototype.drawNodeShape;
LGraphCanvas.prototype.drawNodeShape = function(node, ctx, size, fgcolor, bgcolor, selected, mouse_over) {
    const result = originalDrawNodeShape.apply(this, arguments);
    
    if (node.type === "ColorImage" || node.type === "ColorCode") {
        // Get current color from widget
        const colorWidget = node.widgets?.find(w => w.name === "color");
        const currentColor = node._previewColor || colorWidget?.value || "#FF0000";
        
        // Draw prominent color preview bar at bottom of node
        const previewHeight = 30;  // Taller bar
        const margin = 4;
        const borderRadius = 4;
        
        const x = margin;
        const y = size[1] - previewHeight - margin;
        const width = size[0] - margin * 2;
        const height = previewHeight;
        
        // Draw rounded rectangle with color
        ctx.fillStyle = currentColor;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, width, height, borderRadius);
        } else {
            // Fallback for older browsers
            ctx.rect(x, y, width, height);
        }
        ctx.fill();
        
        // Add subtle border
        ctx.strokeStyle = selected ? "#FFF" : "#666";
        ctx.lineWidth = selected ? 2 : 1;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, width, height, borderRadius);
        } else {
            ctx.rect(x, y, width, height);
        }
        ctx.stroke();
    }
    
    return result;
};
