/**
 * Color Ramp Visual Widget for ComfyUI
 * Creates an interactive gradient editor similar to Blender's ColorRamp
 */

import { app } from "../../scripts/app.js";

console.log("PBR Color Ramp Widget Loading...");

app.registerExtension({
    name: "PBR.ColorRampWidget",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("Checking node:", nodeData.name);
        
        if (nodeData.name === "ColorRamp") {
            console.log("✓ ColorRamp node found, adding widget");
            
            // Override onDrawForeground to draw gradient on node
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                // Set widget start position to leave room for gradient
                // 40 (startY) + 50 (barHeight) + 20 (swatches) + 20 (label) = 130
                this.widgets_start_y = 130;
                
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                
                if (!this.colorStops) return;
                
                try {
                    // Draw gradient bar inside the node, below title and connections
                    const margin = 15;
                    const barHeight = 50;
                    const startY = 40; // Below title bar and connection points
                    const barWidth = this.size[0] - margin * 2;
                    
                    if (barWidth > 0 && this.colorStops.length >= 2) {
                        // Background
                        ctx.fillStyle = "#1a1a1a";
                        ctx.fillRect(margin, startY, barWidth, barHeight);
                        
                        // Create gradient
                        const gradient = ctx.createLinearGradient(margin, startY, margin + barWidth, startY);
                        
                        // Sort stops by position
                        const sortedStops = [...this.colorStops].sort((a, b) => a.pos - b.pos);
                        
                        sortedStops.forEach(stop => {
                            const r = Math.floor(Math.max(0, Math.min(1, stop.r)) * 255);
                            const g = Math.floor(Math.max(0, Math.min(1, stop.g)) * 255);
                            const b = Math.floor(Math.max(0, Math.min(1, stop.b)) * 255);
                            const pos = Math.max(0, Math.min(1, stop.pos));
                            gradient.addColorStop(pos, `rgb(${r}, ${g}, ${b})`);
                        });
                        
                        ctx.fillStyle = gradient;
                        ctx.fillRect(margin, startY, barWidth, barHeight);
                        
                        // Border
                        ctx.strokeStyle = "#888";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(margin, startY, barWidth, barHeight);
                        
                        // Draw stop markers
                        this.colorStops.forEach((stop, idx) => {
                            const x = margin + Math.max(0, Math.min(1, stop.pos)) * barWidth;
                            
                            const isBeingDragged = this.draggingStopIndex === idx;
                            const isSelected = this.selectedStopIndex === idx;
                            
                            // Color swatch square (clickable for color picker)
                            const swatchSize = 16;
                            const swatchY = startY + barHeight + 5;
                            const r = Math.floor(stop.r * 255);
                            const g = Math.floor(stop.g * 255);
                            const b = Math.floor(stop.b * 255);
                            
                            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                            ctx.fillRect(x - swatchSize/2, swatchY, swatchSize, swatchSize);
                            ctx.strokeStyle = isBeingDragged ? "#ff0000" : (isSelected ? "#ffff00" : "#ffffff");
                            ctx.lineWidth = isSelected || isBeingDragged ? 3 : 2;
                            ctx.strokeRect(x - swatchSize/2, swatchY, swatchSize, swatchSize);
                            
                            // Top triangle marker
                            ctx.fillStyle = isBeingDragged ? "#ffff00" : "#ffffff";
                            ctx.beginPath();
                            ctx.moveTo(x, startY);
                            ctx.lineTo(x - 6, startY - 8);
                            ctx.lineTo(x + 6, startY - 8);
                            ctx.closePath();
                            ctx.fill();
                            ctx.strokeStyle = isBeingDragged ? "#ff0000" : "#000000";
                            ctx.lineWidth = isBeingDragged ? 2 : 1.5;
                            ctx.stroke();
                            
                            // Bottom triangle marker
                            ctx.fillStyle = isBeingDragged ? "#ffff00" : "#ffffff";
                            ctx.beginPath();
                            ctx.moveTo(x, startY + barHeight);
                            ctx.lineTo(x - 6, swatchY - 2);
                            ctx.lineTo(x + 6, swatchY - 2);
                            ctx.closePath();
                            ctx.fill();
                            ctx.strokeStyle = isBeingDragged ? "#ff0000" : "#000000";
                            ctx.lineWidth = isBeingDragged ? 2 : 1.5;
                            ctx.stroke();
                        });
                        
                        // Label at bottom
                        ctx.fillStyle = "#aaaaaa";
                        ctx.font = "11px sans-serif";
                        ctx.textAlign = "center";
                        ctx.fillText("Click gradient to add • Double-click marker to remove • Click swatch for color", this.size[0] / 2, startY + barHeight + 35);
                    }
                } catch (e) {
                    console.error("Error drawing gradient:", e);
                }
            };
            
            // Helper to save color stops to widget
            const saveColorStops = function() {
                const widget = this.widgets.find(w => w.name === "color_stops");
                if (widget && this.colorStops) {
                    widget.value = JSON.stringify(this.colorStops);
                }
            };
            
            // Helper to load color stops from widget
            const loadColorStops = function() {
                const widget = this.widgets.find(w => w.name === "color_stops");
                if (widget) {
                    try {
                        this.colorStops = JSON.parse(widget.value);
                    } catch {
                        this.colorStops = [
                            {pos: 0.0, r: 0.0, g: 0.0, b: 0.0},
                            {pos: 1.0, r: 1.0, g: 1.0, b: 1.0}
                        ];
                    }
                }
            };
            
            // Mouse down handler
            nodeType.prototype.onMouseDown = function(e, localPos, graphCanvas) {
                if (!this.colorStops) {
                    loadColorStops.call(this);
                }
                
                const margin = 15;
                const barHeight = 50;
                const startY = 40;
                const barWidth = this.size[0] - margin * 2;
                const hitRadius = 30;
                
                // Check for clicks on existing stops
                for (let i = 0; i < this.colorStops.length; i++) {
                    const stop = this.colorStops[i];
                    const x = margin + stop.pos * barWidth;
                    
                    // Check color swatch click
                    const swatchSize = 16;
                    const swatchY = startY + barHeight + 5;
                    const swatchLeft = x - swatchSize/2;
                    const swatchRight = x + swatchSize/2;
                    const swatchBottom = swatchY + swatchSize;
                    
                    if (localPos[0] >= swatchLeft && localPos[0] <= swatchRight &&
                        localPos[1] >= swatchY && localPos[1] <= swatchBottom) {
                        // Open color picker for this stop
                        this.selectedStopIndex = i;
                        this.openColorPicker(i);
                        if (e.stopPropagation) e.stopPropagation();
                        if (e.preventDefault) e.preventDefault();
                        return true;
                    }
                    
                    // Check marker for dragging
                    const y = startY + barHeight / 2;
                    const dx = localPos[0] - x;
                    const dy = localPos[1] - y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < hitRadius) {
                        this.draggingStopIndex = i;
                        this.selectedStopIndex = i;
                        this.draggingMargin = margin;
                        this.draggingBarWidth = barWidth;
                        if (e.stopPropagation) e.stopPropagation();
                        if (e.preventDefault) e.preventDefault();
                        return true;
                    }
                }
                
                // Check for click on empty gradient (add new stop)
                if (localPos[0] >= margin && localPos[0] <= margin + barWidth &&
                    localPos[1] >= startY && localPos[1] <= startY + barHeight) {
                    // Add new stop at click position
                    const newPos = (localPos[0] - margin) / barWidth;
                    
                    // Find colors to interpolate between
                    const sortedStops = [...this.colorStops].sort((a, b) => a.pos - b.pos);
                    let leftStop = sortedStops[0];
                    let rightStop = sortedStops[sortedStops.length - 1];
                    
                    for (let i = 0; i < sortedStops.length - 1; i++) {
                        if (newPos >= sortedStops[i].pos && newPos <= sortedStops[i + 1].pos) {
                            leftStop = sortedStops[i];
                            rightStop = sortedStops[i + 1];
                            break;
                        }
                    }
                    
                    // Interpolate color
                    const t = (newPos - leftStop.pos) / (rightStop.pos - leftStop.pos);
                    const newColor = {
                        pos: newPos,
                        r: leftStop.r + (rightStop.r - leftStop.r) * t,
                        g: leftStop.g + (rightStop.g - leftStop.g) * t,
                        b: leftStop.b + (rightStop.b - leftStop.b) * t
                    };
                    
                    this.colorStops.push(newColor);
                    this.selectedStopIndex = this.colorStops.length - 1;
                    saveColorStops.call(this);
                    app.graph.setDirtyCanvas(true, true);
                    
                    if (e.stopPropagation) e.stopPropagation();
                    if (e.preventDefault) e.preventDefault();
                    return true;
                }
                
                return false;
            };
            
            // Mouse move handler
            nodeType.prototype.onMouseMove = function(e, localPos, graphCanvas) {
                if (this.draggingStopIndex !== undefined && this.draggingStopIndex !== null) {
                    const x = localPos[0];
                    const newPos = Math.max(0, Math.min(1, (x - this.draggingMargin) / this.draggingBarWidth));
                    
                    this.colorStops[this.draggingStopIndex].pos = newPos;
                    saveColorStops.call(this);
                    
                    if (graphCanvas.setDirty) graphCanvas.setDirty(true, true);
                    if (app.canvas) app.canvas.setDirty(true, true);
                    app.graph.setDirtyCanvas(true, true);
                    
                    if (e.stopPropagation) e.stopPropagation();
                    if (e.preventDefault) e.preventDefault();
                    return true;
                }
                
                // Update cursor
                const margin = 15;
                const barHeight = 50;
                const startY = 40;
                const barWidth = this.size[0] - margin * 2;
                
                // Check if over gradient area
                if (localPos[0] >= margin && localPos[0] <= margin + barWidth &&
                    localPos[1] >= startY && localPos[1] <= startY + barHeight + 25) {
                    graphCanvas.canvas.style.cursor = "crosshair";
                } else {
                    graphCanvas.canvas.style.cursor = "default";
                }
                
                return false;
            };
            
            // Mouse up handler
            nodeType.prototype.onMouseUp = function(e, localPos, graphCanvas) {
                if (this.draggingStopIndex !== undefined && this.draggingStopIndex !== null) {
                    this.draggingStopIndex = null;
                    this.draggingMargin = null;
                    this.draggingBarWidth = null;
                    
                    if (e.stopPropagation) e.stopPropagation();
                    if (e.preventDefault) e.preventDefault();
                    return true;
                }
                
                return false;
            };
            
            // Double-click handler to remove markers
            nodeType.prototype.onDblClick = function(e, localPos, graphCanvas) {
                if (!this.colorStops) {
                    loadColorStops.call(this);
                }
                
                const margin = 15;
                const barHeight = 50;
                const startY = 40;
                const barWidth = this.size[0] - margin * 2;
                const hitRadius = 30;
                
                // Check for double-click on stops to remove them
                for (let i = 0; i < this.colorStops.length; i++) {
                    const stop = this.colorStops[i];
                    const x = margin + stop.pos * barWidth;
                    const y = startY + barHeight / 2;
                    
                    const dx = localPos[0] - x;
                    const dy = localPos[1] - y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < hitRadius) {
                        // Can't remove if only 2 stops left
                        if (this.colorStops.length > 2) {
                            console.log(`Removed color stop at position ${stop.pos.toFixed(2)}`);
                            this.colorStops.splice(i, 1);
                            this.selectedStopIndex = null;
                            saveColorStops.call(this);
                            app.graph.setDirtyCanvas(true, true);
                        } else {
                            console.log("⚠ Cannot remove - need at least 2 color stops");
                        }
                        
                        if (e.stopPropagation) e.stopPropagation();
                        if (e.preventDefault) e.preventDefault();
                        return true;
                    }
                }
                
                return false;
            };
            
            // Color picker helper
            nodeType.prototype.openColorPicker = function(stopIndex) {
                const stop = this.colorStops[stopIndex];
                if (!stop) return;
                
                const r = Math.round(stop.r * 255);
                const g = Math.round(stop.g * 255);
                const b = Math.round(stop.b * 255);
                const hexColor = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
                
                const colorInput = document.createElement('input');
                colorInput.type = 'color';
                colorInput.value = hexColor;
                colorInput.style.position = 'absolute';
                colorInput.style.opacity = '0';
                colorInput.style.pointerEvents = 'none';
                document.body.appendChild(colorInput);
                
                colorInput.addEventListener('change', (e) => {
                    const hex = e.target.value;
                    const r = parseInt(hex.substr(1, 2), 16) / 255;
                    const g = parseInt(hex.substr(3, 2), 16) / 255;
                    const b = parseInt(hex.substr(5, 2), 16) / 255;
                    
                    this.colorStops[stopIndex].r = r;
                    this.colorStops[stopIndex].g = g;
                    this.colorStops[stopIndex].b = b;
                    
                    saveColorStops.call(this);
                    app.graph.setDirtyCanvas(true, true);
                    
                    document.body.removeChild(colorInput);
                });
                
                colorInput.click();
            };
            
            // Node creation
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                console.log("ColorRamp node created");
                
                // Set widget start position to leave room for gradient
                // 40 (startY) + 50 (barHeight) + 20 (swatches) + 20 (label) = 130
                this.widgets_start_y = 130;
                
                // Load initial color stops
                loadColorStops.call(this);
                
                // Hide the color_stops string widget (it's just for data storage)
                const colorStopsWidget = this.widgets.find(w => w.name === "color_stops");
                if (colorStopsWidget) {
                    colorStopsWidget.computeSize = function() { return [0, -4]; };
                }
                
                return result;
            };
            
            console.log("✓ ColorRamp widget setup complete");
        }
    }
});
