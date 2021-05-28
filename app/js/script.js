const model_url = window.location.origin.substr(0,window.location.origin.indexOf(":")+1) + "//" + "modelservice." + window.location.origin.substr(window.location.origin.indexOf(".")+1) + '/model'
//const api_key_1 = "my87udph3c9up73g7nwv3b3pdt1hzf4h";
//const api_key_2 = "mgp9zye1k3rfx7hik70oeaibk4tw08rk";

const model_key_url = window.location.origin + '/model_access_keys'

fetch(model_key_url)
  .then(response => response.json())
  .then(function(data) {
    api_key_1 = data.model_1_access_key
    api_key_2 = data.model_2_access_key
  })

function go_fetch(from_explain=false) {
  d3.select("#loader").attr("style", "display:block");
  let image_path = d3.select("#new_image").node().src
  image_path = image_path.substring(image_path.indexOf("/data")+1)
  let image_data = getBase64Image(document.getElementById('new_image'));
  let post_data_1 = {
    accessKey: api_key_1,
    request: {
      "path" : image_path,
      "image": image_data
    },
  };

  fetch(model_url, {
    method: "POST", // or 'PUT'
    body: JSON.stringify(post_data_1), // data can be `string` or {object}!
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((res) => res.json())
    .then(function (response) {
      d3.select("#prediction_model_1").text(
        response.response.prediction.prediction +
          " (" + response.response.prediction.prediction_value.toFixed(3) + ")"
      );
      if (response.response.prediction.prediction != "normal") {
        let post_data_2 = {
          accessKey: api_key_2,
          request: {
            "path" : image_path,
            "image": image_data
          },
        };

        fetch(model_url, {
          method: "POST",
          body: JSON.stringify(post_data_2),
          headers: {
            "Content-Type": "application/json",
          },
        })
          .then((res) => res.json())
          .then(function (response) {
            d3.select("#prediction_model_2").text(
              response.response.prediction.prediction + " (" + response.response.prediction.prediction_value.toFixed(3) + ")"
            );
            !from_explain ? d3.select("#loader").attr("style", "display:none") : null
          })
          .catch((error) => console.error("Error:", error));
      } else {
        !from_explain ? d3.select("#loader").attr("style", "display:none") : null
      }
    })
    .catch((error) => console.error("Error:", error));
}

function getBase64Image(img) {
  let canvas = document.createElement("canvas");
  let scaling = document.getElementById("new_image").naturalWidth / 1000;
  canvas.width = img.naturalWidth / scaling;
  canvas.height = img.naturalHeight / scaling;
  let ctx = canvas.getContext("2d");
  ctx.drawImage(
    img,0,0,
    img.naturalWidth,
    img.naturalHeight,
    0,0,
    canvas.width,
    canvas.height
  );
  return canvas.toDataURL("image/png");
}

function get_new_image() {
  d3.select("#loader").attr("style", "display:block");
  d3.select("#explain_overlay").attr("src","");
  d3.select("#prediction_model_1").text("...");
  d3.select("#prediction_model_2").text("...");
  fetch(window.location.origin + "/random_image", {
    method: "GET", 
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((res) => res.json())
    .then(function (response) {
      d3.select("#new_image").attr("src", "/" + response.file);
      d3.select("#actual_value").html(
        response.file.substring(
          response.file.indexOf("/") + 6,
          response.file.lastIndexOf("/")
        )
      );
      d3.select("#loader").attr("style", "display:none");
      d3.select("#explain_overlay").attr("style","visibility:hidden");
      d3.select("#image_explain_button").attr("onclick","explain_image();")
    })
    .catch((error) => console.error("Error:", error));
}

const explain_url = window.location.origin + '/explain_image'

function explain_image() {
  if (d3.select("#prediction_model_1").text() == "...") {
    go_fetch(true);
  }
  d3.select("#loader").attr("style", "display:block");
  let image_width = d3.select("#new_image").node().getBoundingClientRect().width
  let image_height = d3.select("#new_image").node().getBoundingClientRect().height
  let image_path = d3.select("#new_image").node().src
  image_path = image_path.substring(image_path.indexOf("/data")+1)
  let explain_image_url = explain_url + "?image=" + image_path
  fetch(explain_image_url)
    .then(response => response.json())
    .then(function(data) {
      d3.select("#explain_overlay").attr("src",data.image);
      d3.select("#explain_overlay").attr("width",image_width);
      d3.select("#explain_overlay").attr("height",image_height);
      d3.select("#loader").attr("style", "display:none");
      d3.select("#explain_overlay").attr("style","visibility:block");
      d3.select("#image_explain_button").attr("onclick","toggle_explained_image();")
  });
}

function toggle_explained_image() {
  if (d3.select("#explain_overlay").attr("style") == "visibility:block") {
    d3.select("#explain_overlay").attr("style","visibility:hidden")
  } else {
    d3.select("#explain_overlay").attr("style","visibility:block");
  }
}